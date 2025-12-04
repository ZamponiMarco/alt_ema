import os
import json
import re
import sys

import joblib
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from libs.qn.examples.closed_queuing_network import example1, example2
from libs.qn.examples.controller import constant_controller
from libs.qn.model.queuing_network import ClosedQueuingNetwork, print_results
from libs.qn.examples.controller import autoscalers
import seaborn as sns

DATA_FOLDER = 'resources/simulation/random/'
PICS_FOLDER = 'resources/pics/'
TRAJECTORY_FOLDER = 'resources/workloads/'
MODEL_FOLDER = 'resources/random_qns/'

INITIAL_CORES = np.array([
    1,
    1,
    1
])

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def plot_results(pd_data, theory, output_folder, output_file):
    plt.figure(figsize=(6, 3.5))
    measured_list = []
    for idx, (key, item) in enumerate(pd_data.items()):
        mean = np.mean(item, axis=0)
        mean = np.concatenate((mean, [mean[-1]]))
        measured, = plt.step(range(len(mean)), mean, color=COLORS[key], linestyle='--', where='post')
        measured_list.append(measured)
        std_error = np.std(item, axis=0) / np.sqrt(len(item))
        std_error = np.concatenate((std_error, [std_error[-1]]))
        plt.fill_between(range(len(mean)), mean - std_error, mean + std_error, color=COLORS[key], alpha=0.2, step='post')
    predicted_list = []
    for i in range(1, theory.shape[1]):
        predicted, = plt.step(list(range(theory.shape[0] + 1)), list(theory[:, i]) + [theory[-1, i]], color=COLORS[i], where='post')
        predicted_list.append(predicted)
    handles = [(measured_list[i], predicted_list[i]) for i in range(len(measured_list))]
    labels = [f'Station {i+1}' for i in range(len(measured_list))]
    plt.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('$t$', fontsize=12)
    plt.ylabel('$\mathbf{s}(t)$', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{output_file}.pdf'))

def process_subfolder(change_interval_folder, subfolder, pattern):
    folder_path = os.path.join(DATA_FOLDER, change_interval_folder, subfolder)
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
    measured_rtvs = np.array([])
    predicted_rtvs = np.array([])
    for simulation_folder in os.listdir(DATA_FOLDER):
        trajectory = json.load(open(os.path.join(TRAJECTORY_FOLDER, f'qn_{simulation_folder}_optimal_load.json')))['users']
        network: ClosedQueuingNetwork = joblib.load(os.path.join(MODEL_FOLDER, f'qn_{simulation_folder}.pkl'))
        network.set_controllers(
            [constant_controller(network, 0, network.max_users)] +
            autoscalers['hpa50'](network)
        )
        
        s, c = network.steady_state_simulation(
            INITIAL_CORES,
            trajectory,
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

        theory_base_response_times = (trajectory / np.array(s[:,0]))- 1/network.mu[0]

        output_folder = os.path.join(PICS_FOLDER, simulation_folder, 'pics')
        os.makedirs(output_folder, exist_ok=True)
        
        try:
            cores_data = process_subfolder(simulation_folder, 'cores', r'Cores\[(\d+)\]')
            arrival_data = process_subfolder(simulation_folder, 'arrival_rates', r'Throughput\[(\d+)\]')
            response_data = process_subfolder(simulation_folder, 'response_times', r'ResponseTime\[(\d+)\]')
            
            s = np.zeros((len(trajectory), network.stations))
            for i in range(len(trajectory)):
                for j in range(network.stations - 1):
                    s[i, j + 1] = arrival_data[j + 1][0][i]
                P = network.probabilities[1:, 0].T
                s[i, 0] = P@s[i, 1:]
            measured_rtv = network.compute_rtv(trajectory, s)
            predicted_rtv = network.compute_rtv(trajectory, theory_arrival_rates)
            
            print("RTV Measured:", measured_rtv)
            print("RTV Predicted:", predicted_rtv)
            measured_rtvs = np.append(measured_rtvs, measured_rtv)
            predicted_rtvs = np.append(predicted_rtvs, predicted_rtv)

            if cores_data:
                plot_results(cores_data, theory_cores, output_folder, f'{simulation_folder}_cores')
            if arrival_data:
                plot_results(arrival_data, theory_arrival_rates, output_folder, f'{simulation_folder}_arrival_rates')
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
                plt.step(range(len(mean_base_responses)), mean_base_responses, color=COLORS[0], linestyle='--', where='post', label='Measured')
                plt.fill_between(range(len(mean_base_responses)), mean_base_responses - std_error_base_responses, mean_base_responses + std_error_base_responses, color=COLORS[0], alpha=0.2, step='post')
                plt.step(list(range(len(theory_base_response_times) + 1)), list(theory_base_response_times) + [theory_base_response_times[-1]], color=COLORS[0], label='Predicted', where='post')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.xlabel('$t$', fontsize=12)
                plt.ylabel('$R(t)$', fontsize=12)
                plt.ylim(0, 1.2 * np.max(mean_base_responses))
                plt.text(0.95, 0.95, f'Measured RTV: {measured_rtv:.2f}\nPredicted RTV: {predicted_rtv:.2f}', transform=plt.gca().transAxes, 
                         verticalalignment='top', horizontalalignment='right', fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f'{simulation_folder}_base_response_times.pdf'))
                
        except Exception as e:
            print(f"Error processing {simulation_folder}: {e}")
            
    if len(measured_rtvs) > 0 and len(predicted_rtvs) > 0:
        print("Measured RTVs:", measured_rtvs)
        print("Predicted RTVs:", predicted_rtvs)
        relative_differences = (measured_rtvs - predicted_rtvs) / measured_rtvs
        print("Relative differences (measured - predicted) / predicted:", relative_differences)
        print(pd.Series(relative_differences).describe())
        
        print("Do the traces cause real failures?")
        
        w_stat, p_value = stats.wilcoxon(measured_rtvs - 20, alternative='greater')
        print(f"Wilcoxon Statistic: {w_stat}")
        print(f"P-value: {p_value:.5e}")
        
        if p_value < 0.05:
            print("\nCONCLUSION: REJECT NULL HYPOTHESIS.")
            print("The measured values are statistically significantly greater than 20.")
            print("The fault is effectively exposed.")
        else:
            print("\nCONCLUSION: FAIL TO REJECT NULL.")
            print("The measured values are not significantly different from 20.")
        
        print("Is under-provisioning estimating conservative?")
        
        w_stat, p_value = stats.wilcoxon(measured_rtvs - predicted_rtvs, alternative='greater')
        print(f"Wilcoxon Statistic: {w_stat}")
        print(f"P-value: {p_value:.5e}")
        
        if p_value < 0.05:
            print("\nCONCLUSION: REJECT NULL HYPOTHESIS.")
            print("The measured values are statistically significantly greater than the predicted ones.")
        else:
            print("\nCONCLUSION: FAIL TO REJECT NULL.")
            print("The measured values are not significantly different from the predicted ones.")
        
        print("Spearman Rank Correlation: Is the prediction correlated with measurement?")
        rho, p_value = stats.spearmanr(predicted_rtvs, measured_rtvs)

        print(f"Spearman Rank Correlation: {rho}")
        print(f"P-value: {p_value}")
        
        # Horizontal seaborn boxplot of relative differences with whiskers at 10th and 90th percentiles
        os.makedirs(PICS_FOLDER, exist_ok=True)
        sns.set_theme(style='whitegrid', context='paper')
        palette = sns.color_palette("vlag", 3)

        rel = np.asarray(relative_differences)
        fig, ax = plt.subplots(figsize=(8, 2.5))

        # flierprops to plot outliers as crosses
        flierprops = dict(marker='x', markeredgecolor=palette[2], markersize=5, linestyle='none', alpha=0.8)

        # boxplot with whiskers at 10th and 90th percentiles, show median line in center
        sns.boxplot(x=rel, orient='h', ax=ax, color=palette[1], width=0.4, whis=[5, 95],
            flierprops=flierprops, medianprops={'color': 'k', 'linewidth': 1.5})

        # zero reference
        ax.axvline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.8)

        ax.set_yticks([])  # single horizontal box, no y ticks needed
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        # increase x-axis tick label fontsize
        ax.tick_params(axis='x', labelsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(PICS_FOLDER, 'relative_differences_boxplot.pdf'), bbox_inches='tight')
        plt.close()
