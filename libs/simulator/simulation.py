import numpy as np
from locust import HttpUser, task
from locust.env import Environment
import random
import time

from libs.simulator.controller import adjust_cpu_quotas
from libs.simulator.docker.docker_infrastructure import DockerNetworkInterface
from libs.qn.model.queuing_network import ClosedQueuingNetwork

class MyUser(HttpUser):

    wait_time = lambda self: np.random.exponential(self.think_time)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_probabilities = self.environment.network.probabilities[0,1:]
        self.think_time = 1/self.environment.network.mu[0]
        self.probabilities = self.environment.network.probabilities[1:,1:]

    @task
    def visit_station_task(self):
        self.station = self.select_index_or_none(self.entry_probabilities)

        while self.station is not None:
            self.client.get(f'/app{self.station + 1}')
            self.station = self.select_index_or_none(self.probabilities[self.station])

    def select_index_or_none(self, probabilities):
        cumulative_probabilities = []
        total = 0

        # Build the cumulative probabilities array
        for p in probabilities:
            total += p
            cumulative_probabilities.append(total)

        # Generate a random number in [0, 1)
        rand = random.random()

        # Check if it falls into any probability range
        for i, cp in enumerate(cumulative_probabilities):
            if rand < cp:
                return i

        # If no index is selected, return None
        return None

class SimulationRunner:
    def __init__(
            self, 
            network: ClosedQueuingNetwork, 
            docker_network: DockerNetworkInterface,
            change_interval: int, 
            monitoring_interval: int,
            core_update: int = 1, 
            spawn_rate=100
            ):
        self.environment = Environment(user_classes=[MyUser], host="http://localhost")
        self.environment.network = network
        self.docker_network = docker_network
        self.change_interval = change_interval
        self.monitoring_interval = monitoring_interval
        self.core_update = core_update
        self.spawn_rate = spawn_rate

    def run(self, user_trajectory):
        env = self.environment
        env.create_local_runner()
        env.runner.start(user_count=0, spawn_rate=1)
        timestamps = []
        ticks_to_update = self.core_update
        for users in user_trajectory:
            if ticks_to_update == 0:
                ticks_to_update = self.core_update
                print("Adjusting CPU quotas")
                adjust_cpu_quotas(
                    env.network.stations, 
                    env.network.controllers, 
                    self.monitoring_interval * self.core_update, 
                    self.docker_network
                    )
            print(f"Changing user count to: {users}")
            env.runner.start(user_count=users, spawn_rate=self.spawn_rate)
            ticks_to_update -= 1
            time.sleep(self.change_interval)
            timestamps.append(time.time())
        env.runner.stop()
        return timestamps