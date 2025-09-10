import os
import json
import re

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import numpy as np

from libs.qn.examples.closed_queuing_network import acmeair_qn, example1, example2
from libs.qn.examples.controller import constant_controller
from libs.qn.model.queuing_network import print_results
from libs.qn.examples.controller import autoscalers

INITIAL_FOLDER = 'resources/simulation/acme'
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

load = json.load(open(os.path.join(TRAJECTORY_FOLDER, 'test.json')))
TRAJECTORY = np.array(load['users'])

s, c = network.steady_state_simulation(
    INITIAL_CORES,
    TRAJECTORY,
    3
)

theory_cores = c[:-1]
new_cores = []
for row in theory_cores:
    new_cores.append(row)
    new_cores.append(row)
    new_cores.append(row)
theory_cores = np.array(new_cores)

theory_arrival_rates = s

theory_base_response_times = (TRAJECTORY / np.array(s[:,0]))-network.mu[0]

def plot_results(pd_data, theory, label_prefix, output_file):
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
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{output_file}.pdf'))

def process_subfolder(change_interval_folder, subfolder, pattern):
    folder_path = os.path.join(INITIAL_FOLDER, change_interval_folder, subfolder)
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
    for change_interval_folder in os.listdir(INITIAL_FOLDER):
        cores_data = process_subfolder(change_interval_folder, 'cores', r'Cores\[(\d+)\]')
        arrival_data = process_subfolder(change_interval_folder, 'arrival_rates', r'Throughput\[(\d+)\]')
        response_data = process_subfolder(change_interval_folder, 'response_times', r'ResponseTime\[(\d+)\]')
        
        runs = len(arrival_data[1]) if arrival_data else 0
        rtvs = np.array([])
        for run in range(runs):
            s = np.zeros((len(TRAJECTORY), network.stations - 1))
            for i in range(len(TRAJECTORY)):
                for j in range(network.stations - 1):
                    s[i, j] = arrival_data[j + 1][run][i]
            rtvs = np.append(rtvs, network.compute_rtv(TRAJECTORY, s))
                
        measured_rtv = rtvs.mean() if rtvs.size > 0 else 0.0
        measured_rtv_std = rtvs.std()
        predicted_rtv = network.compute_rtv(TRAJECTORY, theory_arrival_rates[:, 1:])
        
        print("RTV Measured:", measured_rtv)
        print("RTV Measured Std:", measured_rtv_std)
        print("RTV Predicted:", predicted_rtv)
        
        if cores_data:
            plot_results(cores_data, theory_cores, 'Station', f'{change_interval_folder}_cores')
        if arrival_data:
            plot_results(arrival_data, theory_arrival_rates, 'Station', f'{change_interval_folder}_arrival_rates')
        if response_data:
            base_responses_list = []
            for el in range(len(response_data[1])):
                responses = np.array(list(response_data[key][el] for key in response_data.keys()))
                base_responses_list.append(responses.T@network.visit_vector)
            base_responses = np.array(base_responses_list)
            mean_base_responses = np.mean(base_responses, axis=0)
            std_error_base_responses = np.std(base_responses, axis=0) / np.sqrt(len(base_responses))
            std_error_base_responses = np.concatenate((std_error_base_responses, [std_error_base_responses[-1]]))
            plt.figure(figsize=(6, 3.5))
            mean_base_responses = np.concatenate((mean_base_responses, [mean_base_responses[-1]]))
            plt.step(range(len(mean_base_responses)), mean_base_responses, color=COLORS[0], linestyle='--', where='post', label='Measured ({:.2f} $\pm$ {:.2f})'.format(measured_rtv, measured_rtv_std))
            plt.fill_between(range(len(mean_base_responses)), mean_base_responses - std_error_base_responses, mean_base_responses + std_error_base_responses, color=COLORS[0], alpha=0.2, step='post')
            plt.step(list(range(len(theory_base_response_times) + 1)), list(theory_base_response_times) + [theory_base_response_times[-1]], color=COLORS[0], label='Predicted ({:.2f})'.format(predicted_rtv), where='post')
            leg = plt.legend(fontsize=8)
            leg.set_alpha(0.7)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.xlabel('$t$', fontsize=10)
            plt.ylabel('$r_{\\text{ref}}(t)$', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, f'{change_interval_folder}_base_response_times.pdf'))