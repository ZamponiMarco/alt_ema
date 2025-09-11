import json
import os
import numpy as np
import argparse
from libs.qn.examples.controller import autoscalers
from libs.qn.examples.closed_queuing_network import *
from libs.qn.examples.controller import constant_controller
from libs.qn.model.queuing_network import ClosedQueuingNetwork, print_results

OUTPUT_FOLDER = 'resources/workloads/'

shapes = ['free', 'spike', 'sawtooth', 'ramp']
objectives = ['overprovisioning', 'underprovisioning', 'underprovisioning_time', 'overprovisioning_time']

if __name__ == "__main__":
    network: ClosedQueuingNetwork = acmeair_qn()

    args = argparse.ArgumentParser(description='Get optimal load for a closed queuing network.')
    args.add_argument('--horizon', type=int, default=8, help='Horizon for the simulation.')
    args.add_argument('--cores', type=str, default=','.join(['1'] * (network.stations - 1)), help="Comma-separated list of integers for cores (e.g., 4,1,1,1)")
    args.add_argument('--initial_users', type=int, default=0, help='Initial number of users.')
    args.add_argument('--simulation_ticks_update', type=int, default=1, help='Number of ticks to update the simulation.')
    args.add_argument('--autoscaler', type=str, default='hpa50', choices=autoscalers.keys(), help='Autoscaler to use.')
    args.add_argument('--objective', type=str, default='underprovisioning', choices=objectives, help='Objective to optimize.')
    args.add_argument('--shape', type=str, default='free', choices=shapes, help='Load shape.')
    args.add_argument('--output_file', type=str, default=None, help='Output file for results.')
    args.add_argument('--time_limit', type=int, default=600, help='Time limit for the optimization in seconds.')
    args.add_argument('--tolerance', type=float, help='Tolerance for the optimization objective.')
    args.add_argument('--alpha', type=int, help='Alpha parameter for the optimization objective.')
    args.add_argument('--beta', type=int, help='Beta parameter for the optimization objective.')
    cli_args = args.parse_args()
    
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
    
    options = {
        'objective': cli_args.objective,
        'shape': cli_args.shape,
        'time_limit': cli_args.time_limit,
    }
    
    if cli_args.tolerance is not None:
        options['tol'] = cli_args.tolerance
    if cli_args.alpha is not None:
        options['alpha'] = cli_args.alpha
    if cli_args.beta is not None:
        options['beta'] = cli_args.beta
    
    status, time, solutions, q, c, d_i, s, l, min_q_c = network.model(
        horizon, initial_users, cores, simulation_ticks_update, options
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