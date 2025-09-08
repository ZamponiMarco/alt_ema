import pygad
import numpy as np
import time
import json
import os
from libs.qn.examples.closed_queuing_network import acmeair_qn
from libs.qn.examples.controller import constant_controller, autoscalers
from libs.qn.model.queuing_network import ClosedQueuingNetwork
from libs.qn.pwa.util import build_polytope, is_feasible, hit_and_run
from scipy.optimize import minimize

# Experiment parameters
HORIZON = 30
TIME_LIMIT = 30  # seconds

# Global variables
solutions = []
start_time = None
best_fitness = [-float('inf')]

network: ClosedQueuingNetwork = acmeair_qn()
network.set_controllers(
    [constant_controller(network, 0, network.max_users)] +
    autoscalers['hpa50'](network)
)

cores = [network.max_users] + [1] * (network.stations - 1)
    
horizon = HORIZON
initial_users = 10
simulation_ticks_update = 3

A, b = build_polytope(
    horizon=horizon,
    skewness=network.skewness / simulation_ticks_update,
    l_bounds=(1.0, network.max_users),
    l0=initial_users
)

def on_gen_func(ga_instance):
    global solutions, start_time, best_fitness
    current_best_fitness = ga_instance.best_solution()[1]
    current_time = time.time() - start_time
    
    # Track this generation's best solution
    solutions.append([
        current_time, 
        current_best_fitness,
        ga_instance.generations_completed, 
        ])
    
    if current_best_fitness > best_fitness[0]:
        best_fitness[0] = current_best_fitness
        print(f"New best underprovisioning at generation {ga_instance.generations_completed}: {current_best_fitness}")
    
    # Check if time limit has been exceeded
    if current_time >= TIME_LIMIT:
        print(f"Time limit of {TIME_LIMIT} seconds reached. Stopping GA.")
        return "stop"

def on_mutation(ga_instance, offspring_mutation):
    repaired_population = []
    for individual in offspring_mutation:
        repaired_individual = repair_individual(individual, A, b)
        repaired_population.append(repaired_individual)
    return repaired_population

def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness function for PyGAD.
    """
    return fitness(solution, network, cores, simulation_ticks_update)

def repair_individual(individual, A, b):
    """
    Repair individual to be feasible.
    """
    if is_feasible(individual, A, b):
        return individual
    
    x0 = individual  # Use the original individual as starting point

    # Objective and gradient for 0.5 * ||x - x0||^2
    def obj(x):
        diff = x - x0
        return 0.5 * np.dot(diff, diff)

    def jac(x):
        return x - x0

    constraints = []
    for i in range(A.shape[0]):
        Arow = A[i].copy()
        brow = float(b[i])
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, Arow=Arow, brow=brow: brow - np.dot(Arow, x),
            'jac': lambda x, Arow=Arow: -Arow
        })

    # Run SLSQP to project
    res = minimize(obj,
                    x0,
                    jac=jac,
                    constraints=constraints,
                    method='SLSQP',
                    options={'ftol': 1e-9, 'maxiter': 200, 'disp': False})

    return res.x

def fitness(individual, network, cores, simulation_ticks_update):
    N = np.array(individual)
    s, c = network.steady_state_simulation(
        c_init=cores[1:],
        N=N,
        core_update_ticks=simulation_ticks_update
    )
    if s is None:
        return -float('inf')
    underprovisioning = network.compute_rtv(N, s)
    return underprovisioning

if __name__ == "__main__":
    
    # Initialize population using hit-and-run
    pop_size = 50
    population = []
    x0 = np.array([10.0] * horizon)
    samples = hit_and_run(A, b, x0, n_samples=pop_size, burn_in=10, steps_per_sample=5)
    for sample in samples:
        repair_individual(sample, A, b)
        population.append(sample)

    time_limit = TIME_LIMIT  # seconds
    print("Starting PyGAD GA")
    start_time = time.time()
    best_fitness = [-float('inf')]
    solutions = []  # List to track all solutions
    
    # PyGAD setup
    ga_instance = pygad.GA(
        num_generations=10000,  # Large number, stopped by criteria
        num_parents_mating=20,
        fitness_func=fitness_func,
        initial_population=np.array(population),
        on_generation=on_gen_func,
        on_mutation=on_mutation,
        random_mutation_max_val=network.skewness/simulation_ticks_update,
        random_mutation_min_val=-network.skewness/simulation_ticks_update,
    )
    
    ga_instance.run()
    
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    best_underprovisioning = best_solution_fitness
    elapsed_time = time.time() - start_time
    generation = ga_instance.generations_completed
    print(f"PyGAD completed in {elapsed_time:.2f} seconds, generations: {generation}, best underprovisioning: {best_underprovisioning:.2f}")
    
    # Prepare output structure
    output_data = {
        "time": elapsed_time,
        "solutions": solutions,
        "users": best_solution.tolist() if best_solution is not None else []
    }
    
    # Print the output structure
    print(json.dumps(output_data, indent=4))
    
    # Save the best load
    if best_solution is not None:
        output_dir = 'resources/workloads'
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'test_ga.json'), 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Best load saved to {output_dir}/test_ga.json")
    else:
        print("No feasible loads found.")
    
    