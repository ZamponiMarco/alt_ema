import json
import os
import numpy as np
import argparse
from libs.qn.examples.controller import autoscalers
from libs.qn.examples.closed_queuing_network import example2, example7_sparse
from libs.qn.examples.controller import constant_controller
from libs.qn.model.queuing_network import ClosedQueuingNetwork, print_results

OUTPUT_FOLDER = 'resources/workloads/'

shapes = ['free', 'spike', 'sawtooth', 'ramp']
objectives = ['overprovisioning', 'underprovisioning', 'underprovisioning_time', 'overprovisioning_time']

if __name__ == "__main__":

    args = argparse.ArgumentParser(description='Get optimal load for a closed queuing network.')
    args.add_argument('--horizon', type=int, default=8, help='Horizon for the simulation.')
    args.add_argument('--cores', type=str, default='1,1,1', help="Comma-separated list of integers for cores (e.g., 4,1,1,1)")
    args.add_argument('--initial_users', type=int, default=0, help='Initial number of users.')
    args.add_argument('--simulation_ticks_update', type=int, default=1, help='Number of ticks to update the simulation.')
    args.add_argument('--autoscaler', type=str, default='hpa50', choices=autoscalers.keys(), help='Autoscaler to use.')
    args.add_argument('--objective', type=str, default='underprovisioning', choices=objectives, help='Objective to optimize.')
    args.add_argument('--shape', type=str, default='free', choices=shapes, help='Load shape.')
    args.add_argument('--output_file', type=str, default=None, help='Output file for results.')
    cli_args = args.parse_args()
    
    network: ClosedQueuingNetwork = example2()
    network.set_controllers(
        [constant_controller(network, 0, network.max_users)] +
        autoscalers[cli_args.autoscaler](network)
    )
    
    cores = np.array([network.max_users] + [int(core.strip()) for core in cli_args.cores.split(',')])
    if len(cores) != network.stations:
        raise ValueError(f"Expected {network.stations} cores, got {len(cores)}.")

    horizon = cli_args.horizon
    initial_users = cli_args.initial_users
    simulation_ticks_update = cli_args.simulation_ticks_update

    status, time, solutions, q, c, d_i, s, l, min_q_c = network.model(
        horizon, initial_users, cores, simulation_ticks_update,
        {
            'objective': cli_args.objective,
            'time_limit': 600,
            'shape': cli_args.shape,
        }
    )
    
    output_file = cli_args.output_file
    if output_file:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        file = os.path.join(OUTPUT_FOLDER, output_file)  
        results = {
                "status": status,
                "time": time,
                "solutions": solutions,
                "users": [el.tolist() for el in l] if l is not None else []
            }
        with open(file, 'w') as f:
            json.dump(results, f, indent=4)