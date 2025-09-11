import sys, time, traceback, os, json, argparse
from typing import List, Dict, Any
import numpy as np
from libs.simulator.docker.app.app_info import AppInfo, ContainerInfo
from libs.simulator.docker.docker_infrastructure import ContainerDockerNetwork
from libs.simulator.simulation import SimulationRunner
from libs.qn.examples.closed_queuing_network import acmeair_qn
from libs.qn.examples.controller import constant_controller
from libs.qn.examples.controller import autoscalers

# Configuration Constants
DEFAULT_CHANGE_INTERVAL = 10
DEFAULT_CORE_UPDATE = 1
DEFAULT_OUTPUT_DIR = 'resources/simulation'
WORKLOADS_FILE = 'resources/workloads/samples.json'
WORKLOADS = json.load(open(WORKLOADS_FILE))
DOCKER_IMAGE = 'busy-wait:latest'
STARTUP_DELAY = 10
SHUTDOWN_DELAY = 5
METRICS_TYPES = ['cores', 'arrival_rates', 'response_times']
METRIC_NAMES = {
    'cores': 'Cores',
    'arrival_rates': 'Throughput',
    'response_times': 'ResponseTime'
}

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the load simulator.')
    parser.add_argument('--change_interval', type=int, default=DEFAULT_CHANGE_INTERVAL)
    parser.add_argument('--core_update', type=int, default=DEFAULT_CORE_UPDATE)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--index', type=int, required=True)
    return parser.parse_args()

def setup_docker_network(theoretical_mu: List[float], network_model: Any) -> ContainerDockerNetwork:
    services = [
        ContainerInfo(
            name=f'app{i + 1}',
            image=DOCKER_IMAGE,
            env={
                "DELAY": 1/mu,
                "CORES": int(network_model.max_cores[i + 1])
            },
            path=f'app{i + 1}'
        ) for i, mu in enumerate(theoretical_mu)
    ]
    return ContainerDockerNetwork(AppInfo(requirements=[], services=services))

def get_metric_data(docker_network: ContainerDockerNetwork, timestamps: List[float], 
                   intervals: int, stations: int, metric_type: str) -> Dict[str, List[float]]:
    monitoring_funcs = {
        'cores': lambda ts: docker_network.monitoring.get_current_cores(stations, ts),
        'arrival_rates': lambda ts: docker_network.monitoring.get_arrival_rates(intervals, stations, ts),
        'response_times': lambda ts: docker_network.monitoring.get_response_times(intervals, stations, ts)
    }
    
    data = [monitoring_funcs[metric_type](ts) for ts in timestamps]
    return {
        f"{METRIC_NAMES[metric_type]}[{i + 1}]": [round(float(d[i]), 2) for d in data]
        for i in range(stations)
    }

def collect_and_save_metrics(docker_network: ContainerDockerNetwork, timestamps: List[float], 
                           intervals: int, stations: int, output_dir: str, end_time: float) -> Dict:
    metrics = {
        metric_type: get_metric_data(docker_network, timestamps, intervals, stations, metric_type)
        for metric_type in METRICS_TYPES
    }
    
    for metric_name, data in metrics.items():
        metric_dir = os.path.join(output_dir, f'{intervals}', metric_name)
        os.makedirs(metric_dir, exist_ok=True)
        with open(os.path.join(metric_dir, f'{metric_name}_{end_time}.json'), 'w') as f:
            json.dump(data, f, indent=4)
        print(f"{metric_name} data saved")
    return metrics

def initialize_network_model():
    network_model = acmeair_qn()
    network_model.set_controllers(
        [constant_controller(network_model, 0, network_model.max_users)] +
        autoscalers['hpa50'](network_model)
    )
    return network_model.mu[1:], network_model

def main() -> int:
    args = parse_arguments()
    theoretical_mu, network_model = initialize_network_model()
    docker_network = setup_docker_network(theoretical_mu, network_model)
    runner = SimulationRunner(
        network_model, docker_network, 
        args.change_interval, args.change_interval,
        args.core_update
    )

    try:
        docker_network.run()
        time.sleep(STARTUP_DELAY)

        trajectory = WORKLOADS[args.index]
        print(f"Running simulation with trajectory: {trajectory}")
        timestamps = runner.run(trajectory)

        metrics = collect_and_save_metrics(
            docker_network, timestamps, args.change_interval,
            len(theoretical_mu), args.output_dir, timestamps[-1]
        )
        
        for metric_type, values in metrics.items():
            for key, value in values.items():
                print(f"{key}: {value}")
        return 0

    except KeyboardInterrupt:
        print("Simulation interrupted.")
        return 1
    except Exception as e:
        print(f"An error occurred: [{e}]")
        traceback.print_exc()
        return 2
    finally:
        docker_network.shutdown()
        time.sleep(SHUTDOWN_DELAY)

if __name__ == '__main__':
    raise SystemExit(main())