import os
import matplotlib.pyplot as plt
import numpy as np

from libs.qn.examples.closed_queuing_network import example1, example2
from libs.qn.examples.controller import constant_controller
from libs.qn.model.queuing_network import ClosedQueuingNetwork
from libs.qn.examples.controller import autoscalers

INITIAL_CORES = [
    1,
    1,
    1,
]

TRAJECTORY = [
    10.0,10.0,10.0,10.0,5.0,1.0,1.0,2.5749999999992768,7.575000000000891,12.575000000001051,17.575000000001054,22.575000000001097,27.575000000001054,26.0,21.0,16.0,11.0,6.0,1.0,2.575,7.5749999999999815,12.574999999999982,17.57499999999998,22.57499999999998,27.57499999999998,32.57499999999998,37.574999999999825,42.574999999999825,47.574999999999825,52.574999999999825
]

def plot_queue(transient_q, simulation_ticks):
    time_steps = np.arange(simulation_ticks + 1)

    plt.figure(figsize=(10, 6))

    # Plot transient throughput
    for station in range(transient_q.shape[1]):
        plt.plot(time_steps, transient_q[:, station], label=f"Station {station + 1} (Transient)")

    plt.xlabel("Simulation Ticks")
    plt.ylabel("Queue Length")
    plt.title("Queue Length over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_throughput(transient_s, simulation_ticks):
    """
    Plots the throughput over time from transient simulation and overlays the steady-state solution.

    :param transient_s: 2D numpy array (stations x simulation ticks) containing transient throughput.
    :param steady_s: 1D numpy array (stations) containing steady-state throughput.
    :param simulation_ticks: Number of simulation ticks.
    """
    time_steps = np.arange(simulation_ticks)

    plt.figure(figsize=(10, 6))

    # Plot transient throughput
    for station in range(transient_s.shape[1]):
        plt.plot(time_steps, transient_s[:, station], label=f"Station {station + 1} (Transient)")

    plt.xlabel("Simulation Ticks")
    plt.ylabel("Throughput")
    plt.title("Throughput Over Time (Transient vs. Steady-State)")
    plt.legend()
    plt.grid(True)
    plt.savefig("resources/simulation_throughput.png")


if __name__ == "__main__":
    network: ClosedQueuingNetwork = example2()
    network.set_controllers(
        [constant_controller(network, 0, network.max_users)] +
        autoscalers['hpa50'](network)
    )

    # Steady State Simulation
    q, s, d, c = network.steady_state_simulation(
        len(TRAJECTORY),
        INITIAL_CORES,
        TRAJECTORY,
        3
    )
    print(np.array(s))
    print(np.array(c))
    res_time = np.array(q) / np.array(s)
    print(res_time)
    
    os.makedirs("resources/simulation", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for station in range(1, np.array(s).shape[1]):
        plt.plot(np.array(s)[:, station], label=f"Station {station}")
    plt.xlabel("Simulation Ticks")
    plt.ylabel("Throughput (s)")
    plt.title("Throughput per Station Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("resources/simulation/simulation_throughput.png")

    plt.figure(figsize=(10, 6))
    for station in range(1, np.array(c).shape[1]):
        plt.plot(np.array(c)[:, station], label=f"Station {station}")
    plt.xlabel("Simulation Ticks")
    plt.ylabel("Cores (c)")
    plt.title("Cores per Station Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("resources/simulation/simulation_cores.png")