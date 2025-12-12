import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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

    # Process GA files
    ga_files = glob.glob(os.path.join(DATA_FOLDER, 'test_ga_*.json'))
    for file_path in ga_files:
        data = load_json_file(file_path)

        if 'solutions' in data:
            solutions = data['solutions']
            # Create dataframe with samples and fitness columns
            # samples = 50 * generation
            samples = [50 * sol[2] for sol in solutions]
            fitness = [sol[1] for sol in solutions]  # keep as is, assuming positive
            temp_df = pd.DataFrame({'samples': samples, 'fitness': fitness})
            ga_dfs.append(temp_df)

    # Process Random files
    random_files = glob.glob(os.path.join(DATA_FOLDER, 'test_random_*.json'))
    for file_path in random_files:
        data = load_json_file(file_path)

        if 'solutions' in data:
            solutions = data['solutions']
            # samples = attempt
            samples = [sol[2] for sol in solutions]
            fitness = [sol[1] for sol in solutions]
            temp_df = pd.DataFrame({'samples': samples, 'fitness': fitness})
            random_dfs.append(temp_df)

    return ga_dfs, random_dfs

def plot_combined_summary(ga_dfs, random_dfs, xlim_max=500.0, ax=None, is_inner=False):
    """Plot GA and Random summaries together for comparison.

    Each argument can be a list of DataFrames (runs) or a single DataFrame.
    Returns the figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    # Helper to compute mean curve for a list of runs on that group's own grid
    def mean_curve_for_group(run_dfs):
        # normalize
        if run_dfs is None:
            return None, None
        if isinstance(run_dfs, pd.DataFrame):
            runs = [run_dfs]
        else:
            runs = list(run_dfs)

        # build group's sample grid (union of its runs)
        group_samples = set()
        for df in runs:
            if df is None or len(df) == 0:
                continue
            group_samples.update(df['samples'].values)
            group_samples.add(0.0)
        if not group_samples:
            return None, None
        spoints = np.sort(np.array(list(group_samples)))

        # collect zero-order hold values at spoints for each run
        vals = []
        for df in runs:
            if df is None or len(df) == 0:
                continue
            samples = np.asarray(df['samples'].astype(float).values)
            fitness = np.asarray(df['fitness'].astype(float).values)
            order = np.argsort(samples)
            samples_sorted = samples[order]
            fitness_sorted = (-fitness)[order]
            if samples_sorted.size == 0 or samples_sorted[0] > 0.0:
                samples_sorted = np.concatenate([[0.0], samples_sorted])
                fitness_sorted = np.concatenate([[0.0], fitness_sorted])
            idx = np.searchsorted(samples_sorted, spoints, side='right') - 1
            idx[idx < 0] = 0
            idx[idx >= len(fitness_sorted)] = len(fitness_sorted) - 1
            vals.append(fitness_sorted[idx])

        if not vals:
            return None, None
        mean_vals = np.mean(np.array(vals), axis=0)
        std_vals = np.std(np.array(vals), axis=0)
        return spoints, mean_vals, std_vals

    # Compute mean curves separately
    ga_s, ga_mean, ga_std = mean_curve_for_group(ga_dfs)
    rand_s, rand_mean, rand_std = mean_curve_for_group(random_dfs)

    max_samples = max(
        (ga_s[-1] if ga_s is not None else 0.0),
        (rand_s[-1] if rand_s is not None else 0.0),
    )
    
    ga_s = np.append(ga_s, max_samples)
    rand_s = np.append(rand_s, max_samples)
    ga_mean = np.append(ga_mean, ga_mean[-1])
    rand_mean = np.append(rand_mean, rand_mean[-1])
    ga_std = np.append(ga_std, ga_std[-1])
    rand_std = np.append(rand_std, rand_std[-1])

    # Plot each mean on its own sample grid
    if rand_mean is not None:
        ax.step(rand_s, rand_mean, label='Random', color='lightgreen', where='post')
        ax.fill_between(rand_s, rand_mean - rand_std, rand_mean + rand_std, color='lightgreen', alpha=0.2, step='post')
    if ga_mean is not None:
        ax.step(ga_s, ga_mean, label='Genetic Algorithm', color='lightblue', where='post')
        ax.fill_between(ga_s, ga_mean - ga_std, ga_mean + ga_std, color='lightblue', alpha=0.2, step='post')

    if is_inner:
        ax.axhline(y=20, color='lightcoral', linewidth=0.5)
        
        current_ticks = ax.get_yticks()
        if 20 not in current_ticks:
            new_ticks = np.append(current_ticks, 20)
            new_ticks = np.sort(new_ticks)
            ax.set_yticks(new_ticks)
            # Find the first x where GA mean exceeds 20
            if ga_mean is not None:
                indices = np.where(ga_mean > 20)[0]
                if len(indices) > 0:
                    first_index = indices[0]
                    x_over = ga_s[first_index]
                    # Annotate as a tick mark
                    current_xticks = ax.get_xticks()
                    if x_over not in current_xticks:
                        new_xticks = np.append(current_xticks, x_over)
                        new_xticks = np.sort(new_xticks)
                        ax.set_xticks(new_xticks)
                    ax.axvline(x=x_over, color='lightcoral', linewidth=0.5, linestyle='--', alpha=0.7)
    
    # Set automatic y limit based on data within xlim (0 to 500)
    max_y = -1.0
    if rand_mean is not None:
        visible_indices = rand_s <= xlim_max
        if np.any(visible_indices):
            max_y = max(max_y, np.max((rand_mean + rand_std)[visible_indices]))
    if ga_mean is not None:
        visible_indices = ga_s <= xlim_max
        if np.any(visible_indices):
            max_y = max(max_y, np.max((ga_mean + ga_std)[visible_indices]))
    
    if not is_inner:
        ax.set_xlabel('# Samples', fontsize=12)
        ax.set_ylabel('$|\\rho^{\\ast}_{\\varphi}|$', fontsize=12)
    ax.set_xlim(0.0, xlim_max)
    ax.set_ylim(bottom=-1.0, top=max_y + 5)
    ax.grid(True, linestyle='--', alpha=0.6)
    if not is_inner:
        leg = ax.legend(loc='upper left', fontsize=8)
        leg.set_alpha(0.7)
    fig.tight_layout()
    
    return fig, ax

if __name__ == "__main__":
    # Process the solution data
    ga_dfs, random_dfs = process_solutions_data()

    # Create main plot containing the whole graph
    max_samples = min(
        (ga_dfs[0]['samples'].max() if ga_dfs and not ga_dfs[0].empty else 0.0),
        (random_dfs[0]['samples'].max() if random_dfs and not random_dfs[0].empty else 0.0),
    )
    fig, ax = plot_combined_summary(ga_dfs, random_dfs, xlim_max=max_samples)

    # Create inset for the currently plotted view
    axins = inset_axes(ax, width="30%", height="30%", loc='center right')
    fig2, ax2 = plot_combined_summary(ga_dfs, random_dfs, xlim_max=500.0, ax=axins, is_inner=True)

    # Connect them with red lines signaling the zoom
    axins.set_zorder(0)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red", linewidth=1, alpha=0.5)

    # Save the combined figure
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'combined_summary_samples.pdf'), dpi=300, bbox_inches='tight')
    plt.close()