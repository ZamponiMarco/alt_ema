from libs.qn.examples.closed_queuing_network import acmeair_qn
from libs.qn.examples.controller import constant_controller, autoscalers
from libs.qn.model.queuing_network import ClosedQueuingNetwork
from libs.qn.pwa.util import build_polytope, is_feasible, hit_and_run
import numpy as np
import time
import json
import os

# Experiment parameters
HORIZON = 30
TIME_LIMIT = 30  # seconds

if __name__ == "__main__":
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
    
    # Initialize tracking for the best load
    best_underprovisioning = float('inf')
    best_load = None
    start_time = time.time()
    max_time = TIME_LIMIT  # seconds
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
                underprovisioning,
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
        output_dir = 'resources/workloads'
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'test_random.json'), 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Best load saved to {output_dir}/test_random.json")
    else:
        print("No feasible loads found within the time limit.")
    
    