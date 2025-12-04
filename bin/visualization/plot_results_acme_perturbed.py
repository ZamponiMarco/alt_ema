import os
import json
import re

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from libs.qn.examples.closed_queuing_network import acmeair_qn, example1, example2
from libs.qn.examples.controller import constant_controller
from libs.qn.model.queuing_network import print_results
from libs.qn.examples.controller import autoscalers

SIMULATION_ROOT = 'resources/simulation'
TRAJECTORY_FOLDER = 'resources/workloads'
OUTPUT_FOLDER = 'resources/pics'

# Expanded color palette for better distinction between stations
COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    '#aec7e8',  # light blue
    '#ffbb78',  # light orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
    '#c49c94',  # light brown
    '#f7b6d2',  # light pink
    '#c7c7c7',  # light gray
    '#dbdb8d',  # light olive
    '#9edae5',  # light cyan
]

network = acmeair_qn()
network.set_controllers(
    [constant_controller(network, 0, network.max_users)] +
    autoscalers['hpa50'](network)
)

INITIAL_CORES = np.array([
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
])

def compute_theory_metrics(trajectory):
    s, _ = network.steady_state_simulation(
        INITIAL_CORES,
        trajectory,
        3
    )
    theory_arrival_rates = s
    theory_base_response_times = (trajectory / np.array(s[:, 0])) - network.mu[0]
    return theory_arrival_rates, theory_base_response_times

def plot_results(pd_data, theory, output_file):
    plt.figure(figsize=(6, 3.5))
    # Plot measured (dashed)
    for idx, (key, item) in enumerate(pd_data.items()):
        mean = np.mean(item, axis=0)
        mean = np.concatenate((mean, [mean[-1]]))
        plt.step(range(len(mean)), mean, color=COLORS[key], linestyle='--', where='post')
        std_error = np.std(item, axis=0) / np.sqrt(len(item))
        std_error = np.concatenate((std_error, [std_error[-1]]))
        plt.fill_between(range(len(mean)), mean - std_error, mean + std_error, color=COLORS[key], alpha=0.2, step='post')
    # Plot predicted (solid)
    for i in range(1, theory.shape[1]):
        plt.step(list(range(theory.shape[0] + 1)), list(theory[:, i]) + [theory[-1, i]], color=COLORS[i], where='post')
    # Custom legend: two entries only
    measured_line = plt.Line2D([0], [0], color='black', linestyle='--', label='Measured')
    predicted_line = plt.Line2D([0], [0], color='black', linestyle='-', label='Predicted')
    leg = plt.legend(handles=[measured_line, predicted_line], fontsize=8)
    leg.set_alpha(0.7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('$t$', fontsize=12)
    plt.ylabel('$\mathbf{s}(t)$', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{output_file}.pdf'), dpi=300, bbox_inches='tight')

def process_subfolder(base_folder, change_interval_folder, subfolder, pattern):
    folder_path = os.path.join(base_folder, change_interval_folder, subfolder)
    if os.path.isdir(folder_path):
        pd_data = {}
        for file in os.listdir(folder_path):
            with open(os.path.join(folder_path, file)) as f:
                data = json.load(f)
                for entry in data:
                    match = re.match(pattern, entry)
                    if match:
                        pd_data.setdefault(int(match.group(1)), []).append(data[entry])
        return pd_data 
    return None

if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # Iterate experiments i = 1..30
    total_rtvs = []
    predicted_rtvs = []
    for exp_id in range(1, 31):
        workload_path = os.path.join(TRAJECTORY_FOLDER, f'test_acme_perturbed_{exp_id}.json')
        sim_base_folder = os.path.join(SIMULATION_ROOT, str(exp_id))

        if not (os.path.isfile(workload_path) and os.path.isdir(sim_base_folder)):
            continue

        load = json.load(open(workload_path))
        trajectory = np.array(load['users'])
        theory_arrival_rates, theory_base_response_times = compute_theory_metrics(trajectory)

        arrival_data = process_subfolder(sim_base_folder, '60', 'arrival_rates', r'Throughput\[(\d+)\]')
        response_data = process_subfolder(sim_base_folder, '60', 'response_times', r'ResponseTime\[(\d+)\]')

        rtvs = np.array([])
        s = np.zeros((len(trajectory), network.stations))
        for i in range(len(trajectory)):
            for j in range(network.stations - 1):
                s[i, j + 1] = arrival_data[j + 1][0][i]
            P = network.probabilities[1:, 0].T
            s[i, 0] = P @ s[i, 1:]

        measured_rtv = network.compute_rtv(trajectory, s)
        total_rtvs += [measured_rtv]
        predicted_rtv = network.compute_rtv(trajectory, theory_arrival_rates)
        predicted_rtvs += [predicted_rtv]

        print(f"[exp {exp_id}] RTV Measured:", measured_rtv)
        print(f"[exp {exp_id}] RTV Predicted:", predicted_rtv)

        if arrival_data:
            plot_results(arrival_data, theory_arrival_rates, f'exp_{exp_id}_60_arrival_rates')
        if response_data:
            base_responses_list = []
            # Ensure station key 1 exists
            any_key = next(iter(response_data)) if response_data else None
            if any_key is not None:
                for el in range(len(response_data[any_key])):
                    responses = np.array([response_data[key][el] for key in response_data.keys()])
                    base_responses_list.append(responses.T @ network.visit_vector)
            base_responses = np.array(base_responses_list) if base_responses_list else np.array([])
            if base_responses.size > 0:
                mean_base_responses = np.mean(base_responses, axis=0)
                std_error_base_responses = np.std(base_responses, axis=0) / np.sqrt(len(base_responses))
                std_error_base_responses = np.concatenate((std_error_base_responses, [std_error_base_responses[-1]]))
                plt.figure(figsize=(6, 3.5))
                mean_base_responses = np.concatenate((mean_base_responses, [mean_base_responses[-1]]))
                plt.step(range(len(mean_base_responses)), mean_base_responses, color=COLORS[0], linestyle='--', where='post', label='Measured ({:.2f})'.format(measured_rtv))
                plt.step(list(range(len(theory_base_response_times) + 1)), list(theory_base_response_times) + [theory_base_response_times[-1]], color=COLORS[0], label='Predicted ({:.2f})'.format(predicted_rtv), where='post')
                leg = plt.legend(fontsize=8)
                leg.set_alpha(0.7)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.xlabel('$t$', fontsize=10)
                plt.ylabel('$r_{\\text{ref}}(t)$', fontsize=10)

                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_FOLDER, f'exp_{exp_id}_60_base_response_times.pdf'), dpi=300, bbox_inches='tight')
                plt.close()
    
    rtvs = np.array(total_rtvs)
    predicted_rtvs = np.array(predicted_rtvs)
    
    print(f"MEAN: {rtvs.mean()}")
    print(f"STD. DEV: {rtvs.std()}")
    rtv_threshold = 20.0
    diffs = rtvs - rtv_threshold
    w_stat, p_value = stats.wilcoxon(diffs, alternative='greater')
    print(f"Wilcoxon Statistic: {w_stat}")
    print(f"P-value: {p_value:.5e}")

    if p_value < 0.05:
        print("\nCONCLUSION: REJECT NULL HYPOTHESIS.")
        print("The measured values are statistically significantly greater than 20.")
        print("The fault is effectively exposed.")
    else:
        print("\nCONCLUSION: FAIL TO REJECT NULL.")
        print("The measured values are not significantly different from 20.")
        
    print("Spearman Rank Correlation: Is the prediction correlated with measurement?")
    rho, p_value = stats.spearmanr(predicted_rtvs, rtvs)

    print(f"Spearman Rank Correlation: {rho}")
    print(f"P-value: {p_value}")