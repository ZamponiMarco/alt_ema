from libs.qn.examples.closed_queuing_network import acmeair_qn
from libs.qn.examples.controller import constant_controller, autoscalers
from libs.qn.model.queuing_network import ClosedQueuingNetwork
from libs.qn.pwa.util import build_polytope, is_feasible, hit_and_run
import numpy as np
import time
import json
import os
import argparse

OUTPUT_FOLDER = 'resources/workloads/'

if __name__ == "__main__":
    network: ClosedQueuingNetwork = acmeair_qn()
    network.set_controllers(
        [constant_controller(network, 0, network.max_users)] +
        autoscalers['hpa50'](network)
    )
    
    args = argparse.ArgumentParser(description='Random load generation for a closed queuing network.')
    args.add_argument('--horizon', type=int, default=30, help='Horizon for the simulation.')
    args.add_argument('--cores', type=str, default=','.join(['1'] * (network.stations - 1)), help="Comma-separated list of integers for cores (e.g., 4,1,1,1)")
    args.add_argument('--time_limit', type=int, default=30, help='Time limit for random sampling in seconds.')
    args.add_argument('--initial_users', type=int, default=10, help='Initial number of users.')
    args.add_argument('--output_file', type=str, default='test_random.json', help='Output file for results.')
    cli_args = args.parse_args()
    
    cores = np.array([network.max_users] + [int(core.strip()) for core in cli_args.cores.split(',')])
    
    horizon = cli_args.horizon
    initial_users = cli_args.initial_users
    simulation_ticks_update = 3
    
    A, b = build_polytope(
        horizon=horizon,
        skewness=network.skewness / simulation_ticks_update,
        l_bounds=(1.0, network.max_users),
        l0=initial_users
    )
    
    # Initialize tracking for the best load
    best_underprovisioning = float('inf')
    best_load = None
    start_time = time.time()
    max_time = cli_args.time_limit  # seconds
    attempts = 0  # Counter for random points tried
    solutions = []  # List to track all solutions
    print(f"Starting random sampling with time limit {max_time}s")
    
    while time.time() - start_time < max_time:
        # Generate a random feasible load using hit-and-run sampling
        x0 = np.array([initial_users] * horizon)
        samples = hit_and_run(A, b, x0, n_samples=1, burn_in=10, steps_per_sample=5)
        attempts += 1  # Increment attempts counter
        if len(samples) == 0:
            continue  # Skip if no feasible sample
        
        N = samples[0]  # Use the first sample as the load sequence
        
        # Simulate the load on the network using steady_state_simulation
        s, c = network.steady_state_simulation(
            c_init=cores[1:],  # Exclude the first element (max_users for station 0)
            N=N,
            core_update_ticks=simulation_ticks_update
        )
        
        if s is None:
            continue  # Skip if simulation failed
        
        # Compute underprovisioning objective value
        underprovisioning = network.compute_rtv(N, s)
        
        # Update best if this is better
        if -underprovisioning < best_underprovisioning:
            best_underprovisioning = -underprovisioning
            best_load = N.copy()
            print(f"New best at attempt {attempts}: underprovisioning {best_underprovisioning:.2f}")
            current_time = time.time() - start_time
            solutions.append([
                current_time, 
                -underprovisioning,
                attempts, 
                ])
            
    solutions.append([
        time.time() - start_time, 
        best_underprovisioning,
        attempts, 
    ])
    
    elapsed_time = time.time() - start_time
    print(f"Random sampling completed in {elapsed_time:.2f} seconds, attempts: {attempts}, best underprovisioning: {best_underprovisioning:.2f}")
    
    # Prepare output structure
    output_data = {
        "time": elapsed_time,
        "solutions": solutions,
        "users": best_load.tolist() if best_load is not None else []
    }
    
    # Print the output structure
    print(json.dumps(output_data, indent=4))
    
    # Save the best load to a file
    if best_load is not None:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        with open(os.path.join(OUTPUT_FOLDER, cli_args.output_file), 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Best load saved to {OUTPUT_FOLDER}{cli_args.output_file}")
    else:
        print("No feasible loads found within the time limit.")
    
    