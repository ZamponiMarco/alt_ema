import math
from typing import Optional, List

import cvxpy as cp
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pyscipopt import Model as SCIPModel

from libs.qn.pwa.pwa_function import PWAFunction, gurobi_get_pwa_model
import joblib

solutions = []

def callback_feasible_time(model, where):
    global feas_time
    if where == GRB.Callback.MIPSOL:
        feas_time = model.cbGet(GRB.Callback.RUNTIME)
        solution = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        best_solution = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        if solution < best_solution:
            solutions.append((feas_time, solution))
            print(f"New feasible solution found at time: {feas_time:.2f} seconds, objective: {solution:.2f}")

class ClosedQueuingNetwork:

    def __init__(
            self,
            stations: int,
            probabilities: np.ndarray,
            entry_probabilities: np.ndarray,
            mu: np.ndarray,
            max_cores: np.ndarray,
            min_cores: np.ndarray,
            max_users: int,
            think_time: float,
            skewness: float = 15.0,
    ):
        """
        Initialize a closed queuing network

        :param stations: number of computing stations
        :param probabilities: probability matrix of transitioning between computing stations
        :param entry_probabilities: probability matrix of transitioning from user station to computing stations
        :param mu: service rates per core for each computing station
        :param max_cores: maximum number of cores allocable to each computing station
        :param min_cores: minimum number of cores allocable to each computing station
        :param max_users: maximum amount of users in the system
        :param think_time: average time in seconds between each computing station
        """
        if not np.isclose(entry_probabilities.sum(), 1.0):
            raise ValueError("Entry probabilities must sum to 1.0")

        if not np.all(np.sum(probabilities, axis=1) <= 1.0):
            raise ValueError("Transition probabilities must sum at most to 1.0")

        self.stations: int = stations + 1

        self.probabilities: np.ndarray = np.zeros((self.stations, self.stations))
        self.probabilities[1:, 1:] = probabilities
        self.probabilities[0, 1:] = entry_probabilities
        self.probabilities[1:, 0] = 1 - np.sum(probabilities, axis=1)
        
        self.visit_vector = entry_probabilities @ np.linalg.inv(np.eye(stations) - probabilities)

        self.stoich: np.ndarray = self.probabilities - np.eye(self.stations)

        self.mu: np.ndarray = np.hstack(([think_time], mu))
        self.max_cores: np.ndarray = np.hstack(([max_users], max_cores))
        self.min_cores: np.ndarray = np.hstack(([max_users], min_cores))
        self.max_users: int = max_users
        self.controllers: Optional[List[PWAFunction]] = None

        self.max_disturbance: np.ndarray = np.multiply(self.mu, self.max_cores)
        
        self.skewness = skewness

    def set_controllers(self, controllers: List[PWAFunction]) -> None:
        """
        Set the controllers for the queuing network.

        :param controllers: List of PWAFunction controllers for each station.
        """
        self.controllers = controllers

    def steady_state_gurobi(self, c: list[float], N: float):
        """
        Computes the steady state solution for a closed queuing network given an assignment of cores and
        the amount of users currently using the system

        :param c: amount of cores assigned to each station
        :param N: amount of users in the system
        :return: (q, s) if the problem is feasible, None otherwise
        """

        c = np.hstack(([self.max_users], c))
        model = gp.Model("steady_state")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 10

        q = model.addMVar(self.stations, lb=0, ub=self.max_users, name='queue')
        s = model.addMVar(self.stations, lb=0, name='throughput')
        min_q_c = model.addMVar(self.stations, lb=0, name='min_q_c')  # Auxiliary variable for minimum of q and c

        model.addConstrs((min_q_c[j] == gp.min_((q[j], c[j])) for j in range(self.stations)), name="min_q_c")

        # Throughput
        model.addConstrs((s[j] == self.mu[j] * min_q_c[j] for j in range(self.stations)), name="throughput")

        # Steady-state condition: stoich.T @ s == 0
        for i in range(self.stations):
            model.addConstr(gp.quicksum(self.stoich.T[i, j] * s[j] for j in range(self.stations)) == 0, name=f"steady_{i}")

        # Total users
        model.addConstr(q.sum() == N, name="total_users")

        # Objective: dummy (feasibility)
        model.setObjective(0, GRB.MINIMIZE)

        model.optimize()

        if model.Status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
            print("Problem is infeasible.")
            return None
        else:
            q_val = q.X
            s_val = s.X
            return q_val, s_val

    def steady_state(self, c: list[float], N: int):
        E = 1.0 / np.asarray(self.mu)
        zeta = np.hstack(([1.0], self.visit_vector))
        
        D = float(np.dot(E, zeta))
        rhs = (E[1:] * zeta[1:] / D) * float(N)
        underloaded = np.all(c > rhs)
            
        T = np.empty(self.stations, dtype=float)
        if underloaded:
            scale = float(N) / D
        else:
            scale = float(np.min(c / (E[1:] * zeta[1:])))
        T[:] = zeta * scale
        return T

    def transient_simulation(self, simulation_ticks, delta_t, q_init, c_init, update_ticks):
        ticks = int(simulation_ticks / update_ticks)

        q = np.zeros((simulation_ticks + 1, self.stations))
        q[0] = q_init
        c = np.zeros((ticks + 1, self.stations))
        c[0] = np.hstack(([self.max_users], c_init))
        d = np.zeros((simulation_ticks, self.stations))
        s = np.zeros((simulation_ticks, self.stations))
        min_q_c = np.zeros((simulation_ticks, self.stations))

        for t in range(simulation_ticks):
            tick = int(t / update_ticks)
            tick_start, tick_end = tick * update_ticks, (tick + 1) * update_ticks

            min_q_c[t] = np.minimum(q[t], c[tick])
            s[t] = self.mu * min_q_c[t]  # Element-wise multiplication

            q[t + 1] = q[t] + delta_t * (self.stoich.T @ s[t])

            d[t] = self.probabilities.T @ s[t]

            if t == tick_end - 1:
                for j in range(self.stations):
                    c_val = c[tick][j]
                    avg_disturbance = np.sum(d[tick_start:tick_end, j]) / update_ticks
                    station_data = [c_val, avg_disturbance]
                    c[tick + 1][j] = self.controllers[j](station_data)
            
        return q, s, min_q_c, c

    def steady_state_simulation(self, c_init, N, core_update_ticks):
        """
        Simulate the steady state of a closed queuing network given an initial core allocation and number of users.
        """
        ticks = len(N)
        core_ticks = int(ticks / core_update_ticks)
        
        c = np.zeros((core_ticks + 1, self.stations))
        c[0] = np.hstack(([self.max_users], c_init))

        s = np.zeros((ticks, self.stations))

        for tick in range(ticks):
            current_core_tick = int(tick / core_update_ticks)
            lo = current_core_tick * core_update_ticks
            hi = (current_core_tick + 1) * core_update_ticks
            
            ss_s= self.steady_state(c[current_core_tick,1:], N[tick])
            #print(f"Steady state for tick {tick}: q={ss_q}, s={ss_s}")
            s[tick] = ss_s

            if (tick + 1) % core_update_ticks == 0:
                for j in range(self.stations):
                    c_val = c[current_core_tick][j]
                    avg_disturbance = np.sum(s[lo:hi, j]) / core_update_ticks
                    #print("Avg Disturbance:", c_val)
                    station_data = [c_val, avg_disturbance]
                    c[current_core_tick + 1][j] = self.controllers[j](station_data)

        return s, c

    def model(self, horizon, loads, c_init, simulation_ticks_update, options: dict = {}):
        simulation = False
        if horizon is None:
            horizon = len(loads)
            simulation = True

        num_ticks = int(horizon / simulation_ticks_update)
        
        objective = options.get('objective', 'underprovisioning')
        shape = options.get('shape', 'free')
        tol = options.get('tolerance', 20)
        
        with gp.Env() as env:
            model = gp.Model("optimal_load", env=env)

            # State
            l = model.addMVar(horizon, lb=1, ub=self.max_users, name=f'users')
            c = model.addMVar((num_ticks + 1, self.stations), lb=self.min_cores, ub=self.max_cores, name='cores')

            # Auxiliary Variables
            q = model.addMVar((horizon, self.stations), lb=0, ub=self.max_users, name='queue') # Queue lengths
            min_q_c = model.addMVar((horizon, self.stations), lb=0, name='min_q_c') # Auxiliary variable for minimum 
            s = model.addMVar((horizon, self.stations), lb=0, name='throughput') # Service rates
            d_i = model.addMVar((horizon, self.stations), lb=0, name='internal_disturbance') # Internal per-service Disturbance

            model.update()

            # Initial Conditions
            if simulation:
                model.addConstrs((l[i] == loads[i] for i in range(horizon)), "init_l")
            else:
                if loads != -1:
                    model.addConstr(l[0] == loads, "init_l")
            model.addConstr(c[0] == c_init, "init_c")

            # Application Constraints
            for t in range(horizon):
                tick = int(t / simulation_ticks_update)
                # Throughput
                model.addConstrs((min_q_c[t][j] == gp.min_(q[t, j], c[tick, j]) for j in range(self.stations)), f"min_q_c_{t}")
                model.addConstrs((s[t][j] == self.mu[j] * min_q_c[t][j]) for j in range(self.stations))

                # Internal Disturbance
                model.addConstrs((d_i[t][j] == gp.quicksum(self.probabilities[k][j] * s[t][k] for k in range(self.stations))
                                for j in range(self.stations)), f"disturbance_{t}")

                # Flow Balance
                model.addMConstr(self.stoich.T, s[t], '=', np.zeros(self.stations), name=f'flow_balance_{t}')

                model.addConstr(l[t] == gp.quicksum(q[t][j] for j in range(self.stations)), f"queue_sum_{t}")

            # Core Allocation Updates
            for tick in range(num_ticks):
                tick_start, tick_end = tick * simulation_ticks_update, (tick + 1) * simulation_ticks_update
                for j in range(self.stations):
                    c_val = c[tick][j]
                    avg_disturbance = gp.quicksum(d_i[t][j] for t in range(tick_start, tick_end)) / simulation_ticks_update
                    station_data = [c_val, avg_disturbance]
                    value = gurobi_get_pwa_model(model, self.controllers[j], station_data)
                    model.addConstr(c[tick + 1][j] == value, f"pwa_c_tick{tick}_{j}")

            # Shapes
            objective_mask = np.zeros(horizon)
            if shape == 'free':
                for t in range(1, int(0.15*horizon)):
                    model.addConstr(l[t] == l[t - 1], f"warmup_{t}")
                for t in range(int(0.15*horizon), horizon):
                    model.addConstrs(l[t] - l[t - 1] >= -self.skewness/simulation_ticks_update for t in range(1, horizon))
                    model.addConstrs(l[t] - l[t - 1] <= self.skewness/simulation_ticks_update for t in range(1, horizon))
                objective_mask[:] = 1
            elif shape == 'spike':
                alpha = model.addVar()
                gamma = model.addVar(ub=0.9)
                for t in range(1, int(0.15*horizon)):
                    model.addConstr(l[t] == l[t - 1], f"warmup_{t}")
                for t in range(int(0.15*horizon), int(0.25*horizon)):
                    model.addConstr(l[t] == l[t - 1] + alpha, f"peak_{t}")
                for t in range(int(0.25*horizon), int(0.5*horizon)):
                    model.addConstr(l[t] == gamma * l[t - 1], f"decay_{t}")
                for t in range(int(0.5*horizon), horizon):
                    model.addConstr(l[t] == l[t - 1], f"observation_{t}")
                objective_mask[int(0.15*horizon):horizon] = 1
            elif shape == 'sawtooth':
                num_cycles = 2
                ramp_ratio = 0.6

                cycle_len = int(horizon / (num_cycles + 1))
                ramp_len = int(ramp_ratio * cycle_len)

                for t in range(1, int(cycle_len/2)):
                    model.addConstr(l[t] == l[t - 1], f"warmup_{t}")
                for i in range(num_cycles):
                    base = int(cycle_len/2) + i * cycle_len
                    
                    # Ramp up
                    for t in range(base, base + ramp_len):
                        model.addConstr(l[t] >= l[t - 1] + 1, f"sawtooth_ramp_{t}")
                        model.addConstr(l[t] <= l[t - 1] + self.skewness/simulation_ticks_update, f"sawtooth_ramp_skew_{t}")
                        
                    # Ramp down
                    for t in range(base + ramp_len, base + cycle_len):
                        model.addConstr(l[t] <= l[t - 1] - 1, f"sawtooth_drop_{t}")
                        model.addConstr(l[t] >= l[t - 1] - self.skewness/simulation_ticks_update/ramp_ratio, f"sawtooth_drop_skew_{t}")
                        
                for t in range(int(cycle_len/2) + num_cycles * cycle_len, horizon):
                    model.addConstr(l[t] == l[t - 1], f"observation_{t}")
                objective_mask[int(cycle_len/2):horizon] = 1
            elif shape == 'ramp':
                ramp_len = int(0.25 * horizon)
                alpha = model.addVar(lb=-self.skewness/simulation_ticks_update, ub=self.skewness/simulation_ticks_update, name="alpha")  # slope per time step
                for t in range(1, ramp_len):
                    model.addConstr(l[t] == l[t - 1], f"warmup_{t}")
                for t in range(ramp_len, int(0.75*horizon)):
                    model.addConstr(l[t] == l[t - 1] + alpha, f"ramp_{t}")
                for t in range(int(0.75*horizon), horizon):
                    model.addConstr(l[t] == l[t - 1], f"observation_{t}")
                objective_mask[int(ramp_len):horizon] = 1

            delta = model.addMVar((horizon - 1), lb=-math.inf, name='delta')
            model.addConstrs((delta[t] == l[t + 1] - l[t] for t in range(horizon - 1)), "delta")
            abso = model.addMVar((horizon - 1), name='abs')
            model.addConstrs((abso[t] == gp.abs_(delta[t]) for t in range(horizon - 1)), "absoo")
            penalty = gp.quicksum(abso[t] for t in range(horizon - 1))

            # Objectives
            myM = 1.01 * self.max_users
            if objective == 'underprovisioning' or objective == 'underprovisioning_time':
                underprovisioning_base = model.addMVar((horizon), name='underprovisioning_base') # Underprovisioning base

                for t in range(horizon):
                    under_base_aux = model.addVar(lb=-math.inf)
                    model.addConstr(underprovisioning_base[t] == gp.max_(under_base_aux, 0), f"underprovisioning_base_{t}")
                    model.addConstr(under_base_aux == l[t] - ( self.mu[0] + 1) * q[t][0], "under_base_aux")

                if objective == 'underprovisioning':
                    model.addConstr(gp.quicksum(underprovisioning_base[t] for t in range(horizon)) >= options.get('tol', 20))
                   
                    underprovisioning_base_obj = - gp.quicksum(objective_mask[t] * underprovisioning_base[t] for t in range(horizon))
                    model.setObjectiveN(underprovisioning_base_obj, index=0, weight=options.get('alpha', 4))
                    model.setObjectiveN(penalty, index=1, weight=options.get('beta', 1))
                else:
                    underprovisioning_time = model.addMVar((horizon), name='underprovisioning_time') # Underprovisioning time
                    
                    for t in range(horizon):
                        delta_under = model.addVar(vtype=GRB.BINARY)
                        model.addConstr(-under_base_aux <= myM * (1 - delta_under))
                        model.addConstr(-under_base_aux >= - myM * delta_under)
                        model.addConstr(underprovisioning_time[t] == delta_under, f"underprovisioning_time_{t}")
                            
                    model.addConstr(gp.quicksum(underprovisioning_time[t] for t in range(horizon)) >= 1)
                    
                    underprovisioning_time_obj = - gp.quicksum(objective_mask[t] * underprovisioning_time[t] for t in range(horizon))
                    model.setObjectiveN(underprovisioning_time_obj, index=0, priority=2)
                    model.setObjectiveN(penalty, index=1, priority=1)
            if objective == 'overprovisioning' or objective == 'overprovisioning_time':
                overprovisioning = model.addMVar((horizon, self.stations), name='overprovisioning') # Overprovisioning
                
                for t in range(horizon):
                    over_aux = model.addMVar((self.stations), lb=-math.inf)
                    model.addConstrs((over_aux[j] == c[tick][j] - 2 * s[t][j] / self.mu[j] - 1 for j in range(self.stations)))
                    model.addConstrs((overprovisioning[t][j] == gp.max_(over_aux[j], 0) for j in range(self.stations)), f"overprovisioning_{t}")
                
                if objective == 'overprovisioning':
                    model.addConstr(gp.quicksum(overprovisioning[t][j] for t in range(horizon) for j in range(1, self.stations)) >= options.get('tol', 20))
                    
                    overprovisioning_obj = - gp.quicksum(objective_mask[t] * overprovisioning[t][j] for t in range(horizon) for j in range(1, self.stations))
                    model.setObjectiveN(overprovisioning_obj, index=0, weight=options.get('alpha', 10))
                    model.setObjectiveN(penalty, index=1, weight=options.get('beta', 1))
                else:
                    overprovisioning_time = model.addMVar((horizon), name='overprovisioning_time') # Overprovisioning time
                    
                    for t in range(horizon):
                        delta_over = model.addVar(vtype=GRB.BINARY)
                        model.addConstr(-gp.quicksum(over_aux[j] for j in range(self.stations)) <= myM * (1 - delta_over))
                        model.addConstr(-gp.quicksum(over_aux[j] for j in range(self.stations)) >= - myM * delta_over)
                        model.addConstr(overprovisioning_time[t] == delta_over, f"overprovisioning_time_{t}")
                    
                    model.addConstr(gp.quicksum(overprovisioning_time[t] for t in range(horizon)) >= 1)
                    
                    overprovisioning_time_obj = - gp.quicksum(objective_mask[t] * overprovisioning_time[t] for t in range(horizon))
                    model.setObjectiveN(overprovisioning_time_obj, index=0, priority=2)
                    model.setObjectiveN(penalty, index=1, priority=1)
            
            model.setParam("TimeLimit", options.get("time_limit", 60))

            # Solve
            model.optimize(callback=callback_feasible_time)
            
            # Handle Solver Output
            try:
                q_values = np.array([[q[t][j].X for j in range(self.stations)] for t in range(horizon)])
                c_values = np.array([[c[tick][j].X for j in range(self.stations)] for tick in range(num_ticks + 1)])
                d_i_values = np.array([[d_i[t][j].X for j in range(self.stations)] for t in range(horizon)])
                s_values = np.array([[s[t][j].X for j in range(self.stations)] for t in range(horizon)])
                l_values = np.array([l[t].X for t in range(horizon)])
                min_q_c_values = np.array([[min_q_c[t][j].X for j in range(self.stations)] for t in range(horizon)])
            except Exception:
                q_values = c_values = d_i_values = s_values = l_values = min_q_c_values = None
                
            return model.status, model.Runtime, solutions, q_values, c_values, d_i_values, s_values, l_values, min_q_c_values

    def compute_rtv(self, l: np.ndarray, s: np.ndarray) -> float:
        ticks = len(l)
        rtv = 0.0
        for t in range(ticks):
            q0 = s[t][0] / self.mu[0]
            rtv += max(l[t] - ( self.mu[0] + 1) * q0, 0)
        return rtv
        
    def __str__(self):
        controllers_str = ""
        if self.controllers is not None:
            controllers_str = f"Controllers: {[str(ctrl) for ctrl in self.controllers]}"
        else:
            controllers_str = "Controllers: None"
        return (f"Closed Queuing Network with {self.stations - 1} stations:\n"
                f"Probabilities:\n{self.probabilities}\n"
                f"Visit Vector: {self.visit_vector}\n"
                f"Service Rates: {self.mu}\n"
                f"Max Cores: {self.max_cores}\n"
                f"Min Cores: {self.min_cores}\n"
                f"Max Users: {self.max_users}\n"
                f"Skewness: {self.skewness}\n"
                f"{controllers_str}")
       
    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str):
        return joblib.load(filepath)
             
def print_results(network, q, c, s, l):
        def format_data(label, data, index_start, index_end):
            lines = []
            for j in range(index_start, index_end):
                formatted = ",\t".join(f"{row[j]:.2f}" for row in data)
                lines.append(f"{label}[{j}]:\t [\t{formatted}\t]")
            return "\n".join(lines)

        # Print Users
        print("Users:\t\t [\t" + ",\t".join(f"{val:.2f}" for val in l) + "\t]")

        # Print Queues, Bottlenecks, Cores, Throughput, Underprovisioning, and Overprovisioning
        print(format_data("Queue", q, 0, network.stations))
        print(format_data("Cores", c, 1, network.stations))
        print(format_data("Throughput", s, 0, network.stations))