import json
import os
import numpy as np
from libs.qn.examples.controller import autoscalers
from libs.qn.examples.controller import constant_controller
from libs.qn.model.queuing_network import ClosedQueuingNetwork

OUTPUT_FOLDER = 'resources/workloads/'
QN_FOLDER = 'resources/random_qns/'

if __name__ == "__main__":
    
    for file in os.listdir(QN_FOLDER):
        network: ClosedQueuingNetwork = ClosedQueuingNetwork.load(os.path.join(QN_FOLDER, file))
        network.set_controllers(
            [constant_controller(network, 0, network.max_users)] +
            autoscalers['hpa50'](network)
        )

        cores = network.min_cores
        if len(cores) != network.stations:
            raise ValueError(f"Expected {network.stations} cores, got {len(cores)}.")

        horizon = 24
        initial_users = 1
        simulation_ticks_update = 3

        status, time, solutions, q, c, d_i, s, l, min_q_c = network.model(
            horizon, initial_users, cores, simulation_ticks_update,
            {
                'objective': 'underprovisioning',
                'time_limit': 3600,
                'shape': 'sawtooth',
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