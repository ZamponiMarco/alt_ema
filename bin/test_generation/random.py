import json
import os
import argparse
import numpy as np
from libs.qn.examples.controller import autoscalers
from libs.qn.examples.controller import constant_controller
from libs.qn.model.queuing_network import ClosedQueuingNetwork

OUTPUT_FOLDER = 'resources/workloads/'
QN_FOLDER = 'resources/random_qns/'

shapes = ['free', 'spike', 'sawtooth', 'ramp']
objectives = ['overprovisioning', 'underprovisioning', 'underprovisioning_time', 'overprovisioning_time']

if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description='Generate optimal loads for random queuing networks.')
    args.add_argument('--qn-folder', type=str, default=QN_FOLDER, help='Folder containing the queuing networks.')
    args.add_argument('--output-folder', type=str, default=OUTPUT_FOLDER, help='Output folder for the workloads.')
    args.add_argument('--horizon', type=int, default=24, help='Horizon for the simulation.')
    args.add_argument('--initial_users', type=int, default=1, help='Initial number of users.')
    args.add_argument('--objective', type=str, default='underprovisioning', choices=objectives, help='Objective to optimize.')
    args.add_argument('--shape', type=str, default='sawtooth', choices=shapes, help='Load shape.')
    args.add_argument('--time_limit', type=int, default=3600, help='Time limit for the optimization in seconds.')
    cli_args = args.parse_args()
    
    QN_FOLDER = cli_args.qn_folder
    OUTPUT_FOLDER = cli_args.output_folder
    
    for file in os.listdir(QN_FOLDER):
        network: ClosedQueuingNetwork = ClosedQueuingNetwork.load(os.path.join(QN_FOLDER, file))
        network.set_controllers(
            [constant_controller(network, 0, network.max_users)] +
            autoscalers['hpa50'](network)
        )

        cores = network.min_cores
        if len(cores) != network.stations:
            raise ValueError(f"Expected {network.stations} cores, got {len(cores)}.")

        status, time, solutions, q, c, d_i, s, l, min_q_c = network.model(
            cli_args.horizon, cli_args.initial_users, cores, 3,
            {
                'objective': cli_args.objective,
                'time_limit': cli_args.time_limit,
                'shape': cli_args.shape,
            }
        )
        
        output_file = f"{file.split('.')[0]}_optimal_load.json"
        if output_file:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            file = os.path.join(OUTPUT_FOLDER, output_file)  
            results = {
                    "status": status,
                    "time": time,
                    "users": [el.tolist() for el in l] if l is not None else []
                }
            with open(file, 'w') as f:
                json.dump(results, f, indent=4)