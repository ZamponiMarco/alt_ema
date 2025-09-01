import time
from locust import FastHttpUser, constant, task
from locust.env import Environment
import numpy as np
from tqdm import tqdm
from libs.simulator.docker.app.app_info import AppInfo, ContainerInfo
from libs.simulator.docker.docker_infrastructure import ContainerDockerNetwork, ServiceDockerNetwork
from libs.qn.examples.closed_queuing_network import example1_system_mu, example2, example2_system_mu
from libs.qn.model.queuing_network import ClosedQueuingNetwork

class EstimatorUser(FastHttpUser):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.station = self.environment.station

    @task
    def visit_station_task(self):
        self.client.get(f'/app{self.station + 1}')

def sleep_with_progress(x):
    for _ in tqdm(range(x), desc="Sleeping", unit="s"):
        time.sleep(1)

if __name__ == '__main__':
    try:
        mu: np.ndarray = example2_system_mu()
        model: ClosedQueuingNetwork = example2()
        app_info = AppInfo(
            requirements=[],
            services=[
                ContainerInfo(
                    name=f'app{i + 1}',
                    image='busy-wait:latest',
                    env={
                        "DELAY": 1/mu[i],
                        "CORES": int(model.max_cores[i + 1]),
                    },
                    path=f'app{i + 1}',
                ) for i in range(len(mu))
            ] 
        )

        docker_network = ContainerDockerNetwork(app_info)
        docker_network.run()
        
        test_length = 1200
        
        time.sleep(10)
        
        print("Starting simulation")
        
        environments = []
        
        for i in range(len(mu)):
            environment = Environment(user_classes=[EstimatorUser], host="http://localhost")
            environment.create_local_runner()
            environment.station = i
            environment.runner.start(user_count=1, spawn_rate=1)
            environments.append(environment)

        sleep_with_progress(test_length)
        end_time = time.time()
        for environment in environments:
            environment.runner.stop()
        
        print("Simulation finished")
        
        rates = docker_network.monitoring.get_arrival_rates(test_length - 20, len(mu), end_time)
        think_times = docker_network.monitoring.get_current_think_times(test_length - 20, 20, len(mu), end_time)
        response_times = docker_network.monitoring.get_response_times(test_length - 20, len(mu), end_time)
        print(f"Arrival rates: {rates}")
        print(f"Think times: {think_times}")
        print(f"Response times: {response_times}")
    except Exception as e:
        print(f"An error occurred: [{e}]")
    finally:
        docker_network.shutdown()
        time.sleep(5)