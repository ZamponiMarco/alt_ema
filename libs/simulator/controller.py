import numpy as np

from libs.simulator.docker.docker_infrastructure import DockerNetworkInterface
import requests

def adjust_cpu_quotas(stations: int, controllers, monitoring_period, docker_network: DockerNetworkInterface):

    # monitoring and analysis
    current_cores = docker_network.monitoring.get_current_cores(stations - 1)
    arrival_rates = docker_network.monitoring.get_arrival_rates(monitoring_period, stations - 1)
    
    print(f"Avg Cores {monitoring_period}s: {current_cores}")
    print(f"Avg Arrival Rates {monitoring_period}s: {arrival_rates}")

    # planning
    new_cores = []
    for station in range(1, stations):
        i = station - 1
        input_data = np.array([
            current_cores[i],
            arrival_rates[i]
        ])
        new_cores.append(controllers[station](input_data))
    
    print(f"New Cores: {new_cores}")

    # execution
    for entry in range(len(new_cores)):
        container_name = f"app{entry + 1}"
        docker_network.controller.set_resources(container_name, new_cores[entry])
        
def set_cpu_quotas(cores, docker_network):
    # execution
    for entry in range(len(cores)):
        container_name = f"app{entry + 1}"
        docker_network.controller.set_resources(container_name, cores[entry])
        
    for entry in range(len(cores)):
        app_url = f"http://localhost/app{entry + 1}/set_cores"
        try:
            response = requests.post(app_url, json={'cores': cores[entry]}, timeout=2)
            if response.status_code == 200:
                print(f"Set cores for app{entry + 1}: {cores[entry]}")
            else:
                print(f"Failed to set cores for app{entry + 1}: {response.text}")
        except requests.RequestException as e:
            print(f"Error setting cores for app{entry + 1}: {e}")