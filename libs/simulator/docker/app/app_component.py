from abc import ABC, abstractmethod

from libs.simulator.docker.app.app_info import ContainerInfo


class AppComponent(ABC):
    def __init__(self, docker_network, container_info: ContainerInfo):
        self.docker_network = docker_network
        self.container_info = container_info

    @abstractmethod
    def start(self):
        pass