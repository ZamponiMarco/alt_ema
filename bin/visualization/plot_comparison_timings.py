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

    # Process Opt Files
    opt_files = glob.glob(os.path.join(DATA_FOLDER, 'test_opt_*.json'))
    for file_path in opt_files:
        data = load_json_file(file_path)
        
        if 'solutions' in data:
            solutions = data['solutions']
            # Create dataframe with only time and fitness columns
            temp_df = pd.DataFrame(solutions, columns=['time', 'fitness'])
            test_dfs.append(temp_df[['time', 'fitness']])

    return ga_dfs, test_dfs

def plot_combined_summary(ga_dfs, opt_dfs, output_name='combined_summary.png'):
    """Plot GA, Random and Test summaries together for comparison.

    Each argument can be a list of DataFrames (runs) or a single DataFrame.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))

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
    test_t, test_mean, test_std = mean_curve_for_group(opt_dfs)

    max_time = max(
        (ga_t[-1] if ga_t is not None else 0.0),
        (test_t[-1] if test_t is not None else 0.0)
    )
    
    ga_t = np.append(ga_t, max_time)
    test_t = np.append(test_t, max_time)
    ga_mean = np.append(ga_mean, ga_mean[-1])
    test_mean = np.append(test_mean, test_mean[-1])
    ga_std = np.append(ga_std, ga_std[-1])
    test_std = np.append(test_std, test_std[-1])

    # Plot each mean on its own time grid
    if ga_mean is not None:
        ax.step(ga_t, ga_mean, label='Genetic Algorithm', color='lightblue', where='post')
        ax.fill_between(ga_t, ga_mean - ga_std, ga_mean + ga_std, color='lightblue', alpha=0.2, step='post')
    if test_mean is not None:
        ax.step(test_t, test_mean, label='MILP', color='lightcoral', where='post')
        ax.fill_between(test_t, test_mean - test_std, test_mean + test_std, color='lightcoral', alpha=0.2, step='post')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Best $- \\rho_{\\varphi}$', fontsize=12)
    ax.set_xlim(0.0, 300)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle='--', alpha=0.6)
    leg = ax.legend(fontsize=8)
    leg.set_alpha(0.7)
    fig.tight_layout()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_FOLDER, output_name), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    network: ClosedQueuingNetwork = acmeair_qn()
    network.set_controllers(
        [constant_controller(network, 0, network.max_users)] +
        autoscalers['hpa50'](network)
    )

    # Process the solution data
    ga_dfs, test_dfs = process_solutions_data()

    plot_combined_summary(ga_dfs, test_dfs, output_name='combined_summary_timings.pdf')