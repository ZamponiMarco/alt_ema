import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import glob
from libs.qn.examples.controller import constant_controller, autoscalers
from libs.qn.model.queuing_network import ClosedQueuingNetwork
from libs.qn.examples.closed_queuing_network import acmeair_qn

DATA_FOLDER = 'resources/workloads/'
OUTPUT_FOLDER = 'resources/pics/'  

def load_json_file(filepath):
    """Load a JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)

def process_solutions_data():
    """Process all solution files and create arrays of dataframes for each run."""

    # Initialize arrays to hold dataframes for each run
    ga_dfs = []
    random_dfs = []
    test_dfs = []

    # Process GA files
    ga_files = glob.glob(os.path.join(DATA_FOLDER, 'test_ga_*.json'))
    for file_path in ga_files:
        data = load_json_file(file_path)

        if 'solutions' in data:
            solutions = data['solutions']
            # Create dataframe with only time and fitness columns
            temp_df = pd.DataFrame(solutions, columns=['time', 'fitness', 'generation'])
            temp_df = temp_df[['time', 'fitness']]  # Keep only time and fitness
            ga_dfs.append(temp_df)

    # Process Random files
    random_files = glob.glob(os.path.join(DATA_FOLDER, 'test_random_*.json'))
    for file_path in random_files:
        data = load_json_file(file_path)

        if 'solutions' in data:
            solutions = data['solutions']
            # Create dataframe with only time and fitness columns
            temp_df = pd.DataFrame(solutions, columns=['time', 'fitness', 'attempt'])
            temp_df = temp_df[['time', 'fitness']]  # Keep only time and fitness
            random_dfs.append(temp_df)

    # Process Opt Files
    opt_files = glob.glob(os.path.join(DATA_FOLDER, 'test_opt_*.json'))
    for file_path in opt_files:
        data = load_json_file(file_path)
        
        if 'solutions' in data:
            solutions = data['solutions']
            # Create dataframe with only time and fitness columns
            temp_df = pd.DataFrame(solutions, columns=['time', 'fitness'])
            test_dfs.append(temp_df[['time', 'fitness']])

    return ga_dfs, random_dfs, test_dfs

def plot_random_runs_summary(dfs, output_name='runs_summary.png', title=None):
    """Generic summary plot for runs using zero-order hold alignment.

    - dfs: list of DataFrames (each with columns ['time','fitness']) or a single DataFrame
    - output_name: filename under OUTPUT_FOLDER
    - title: optional title
    """
    # Normalize input
    if dfs is None:
        return
    if isinstance(dfs, pd.DataFrame):
        run_dfs = [dfs]
    else:
        run_dfs = list(dfs)

    # Collect all unique time points across runs
    all_times = set()
    for df in run_dfs:
        if df is None or len(df) == 0:
            continue
        all_times.update(df['time'].values)
        all_times.add(0.0)
    if not all_times:
        return
    time_points = np.sort(np.array(list(all_times)))

    # Build zero-order hold values at time_points
    all_fitness_at_time = []
    for df in run_dfs:
        if df is None or len(df) == 0:
            continue
        times = df['time'].values
        fitness = df['fitness'].values

        # Ensure sorted and positive fitness
        order = np.argsort(times)
        times_sorted = times[order]
        fitness_sorted = (-fitness)[order]

        # Prepend 0 if needed
        if times_sorted[0] > 0.0:
            times_sorted = np.concatenate([[0.0], times_sorted])
            fitness_sorted = np.concatenate([[0.0], fitness_sorted])

        idx = np.searchsorted(times_sorted, time_points, side='right') - 1
        idx[idx < 0] = 0
        idx[idx >= len(fitness_sorted)] = len(fitness_sorted) - 1
        interp_vals = fitness_sorted[idx]
        all_fitness_at_time.append(interp_vals)

    if not all_fitness_at_time:
        return

    all_fitness_at_time = np.array(all_fitness_at_time)
    mean_fitness = np.mean(all_fitness_at_time, axis=0)
    std_dev = np.std(all_fitness_at_time, axis=0)

    plt.figure(figsize=(6, 3.5))
    plt.plot(time_points, mean_fitness, 'b-', linewidth=2, label='Mean')
    plt.fill_between(time_points, mean_fitness - std_dev, mean_fitness + std_dev,
                     alpha=0.3, color='blue', label='Mean ± 1σ')
    for vals in all_fitness_at_time:
        plt.step(time_points, vals, where='post', color='gray', alpha=0.1)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Fitness (Positive Underprovisioning)')
    plt.title(title or 'Runs Summary')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_FOLDER, output_name), dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_summary(ga_dfs, random_dfs, opt_dfs, output_name='combined_summary.png'):
    """Plot GA, Random and Test summaries together for comparison.

    Each argument can be a list of DataFrames (runs) or a single DataFrame.
    """
    plt.figure(figsize=(6, 3.5))

    # Helper to compute mean curve for a list of runs on that group's own grid
    def mean_curve_for_group(run_dfs):
        # normalize
        if run_dfs is None:
            return None, None
        if isinstance(run_dfs, pd.DataFrame):
            runs = [run_dfs]
        else:
            runs = list(run_dfs)

        # build group's time grid (union of its runs)
        group_times = set()
        for df in runs:
            if df is None or len(df) == 0:
                continue
            group_times.update(df['time'].values)
            group_times.add(0.0)
        if not group_times:
            return None, None
        tpoints = np.sort(np.array(list(group_times)))

        # collect zero-order hold values at tpoints for each run
        vals = []
        for df in runs:
            if df is None or len(df) == 0:
                continue
            times = np.asarray(df['time'].astype(float).values)
            fitness = np.asarray(df['fitness'].astype(float).values)
            order = np.argsort(times)
            times_sorted = times[order]
            fitness_sorted = (-fitness)[order]
            if times_sorted.size == 0 or times_sorted[0] > 0.0:
                times_sorted = np.concatenate([[0.0], times_sorted])
                fitness_sorted = np.concatenate([[0.0], fitness_sorted])
            idx = np.searchsorted(times_sorted, tpoints, side='right') - 1
            idx[idx < 0] = 0
            idx[idx >= len(fitness_sorted)] = len(fitness_sorted) - 1
            vals.append(fitness_sorted[idx])

        if not vals:
            return None, None
        mean_vals = np.mean(np.array(vals), axis=0)
        std_vals = np.std(np.array(vals), axis=0)
        return tpoints, mean_vals, std_vals

    # Compute mean curves separately
    ga_t, ga_mean, ga_std = mean_curve_for_group(ga_dfs)
    rand_t, rand_mean, rand_std = mean_curve_for_group(random_dfs)
    test_t, test_mean, test_std = mean_curve_for_group(opt_dfs)

    max_time = max(
        (ga_t[-1] if ga_t is not None else 0.0),
        (rand_t[-1] if rand_t is not None else 0.0),
        (test_t[-1] if test_t is not None else 0.0)
    )
    
    ga_t = np.append(ga_t, max_time)
    rand_t = np.append(rand_t, max_time)
    test_t = np.append(test_t, max_time)
    ga_mean = np.append(ga_mean, ga_mean[-1])
    rand_mean = np.append(rand_mean, rand_mean[-1])
    test_mean = np.append(test_mean, test_mean[-1])
    ga_std = np.append(ga_std, ga_std[-1])
    rand_std = np.append(rand_std, rand_std[-1])
    test_std = np.append(test_std, test_std[-1])

    # Plot each mean on its own time grid
    if rand_mean is not None:
        plt.plot(rand_t, rand_mean, label='Random', color='lightgreen')
        plt.fill_between(rand_t, rand_mean - rand_std, rand_mean + rand_std, color='lightgreen', alpha=0.2)
    if ga_mean is not None:
        plt.plot(ga_t, ga_mean, label='GA', color='lightcoral')
        plt.fill_between(ga_t, ga_mean - ga_std, ga_mean + ga_std, color='lightcoral', alpha=0.2)
    if test_mean is not None:
        plt.plot(test_t, test_mean, label='Opt', color='lightblue')
        plt.fill_between(test_t, test_mean - test_std, test_mean + test_std, color='lightblue', alpha=0.2)

    plt.xlabel('Time (seconds)')
    plt.ylabel('RTV')
    plt.xlim(0.0, max_time)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_FOLDER, output_name), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    network: ClosedQueuingNetwork = acmeair_qn()
    network.set_controllers(
        [constant_controller(network, 0, network.max_users)] +
        autoscalers['hpa50'](network)
    )

    # Process the solution data
    ga_dfs, random_dfs, test_dfs = process_solutions_data()

    # Plot the random runs
    if random_dfs:
        print("\nGenerating plots for random runs...")
        plot_random_runs_summary(random_dfs, output_name='random_runs_summary.png',
                                 title='Random Search Summary - Mean Fitness with Deviation')
        
    if ga_dfs:
        print("\nGenerating plots for GA runs...")
        plot_random_runs_summary(ga_dfs, output_name='ga_runs_summary.png',
                                 title='Genetic Algorithm Summary - Mean Fitness with Deviation')
        
    if test_dfs:
        print("\nGenerating plot for test run...")
        plot_random_runs_summary(test_dfs, output_name='test_run.png',
                                 title='Test Run - Fitness Over Time')

    plot_combined_summary(ga_dfs, random_dfs, test_dfs, output_name='combined_summary.png')