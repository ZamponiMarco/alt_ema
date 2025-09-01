from abc import ABC, abstractmethod
import docker

class DockerController(ABC):
    def __init__(self, client: docker.DockerClient):
        self.client = client
    
    @abstractmethod
    def set_resources(self, container_name: str, cores: float):
        pass

class ContainerController(DockerController):
    def __init__(self, client: docker.DockerClient):
        super().__init__(client)
    
    def set_resources(self, container_name: str, num_cores: float):
        new_quota = int(num_cores * 100_000)
        container = self.client.containers.get(container_name)
        container.update(cpu_quota=new_quota)

class ServiceController(DockerController):
    def __init__(self, client: docker.DockerClient):
        super().__init__(client)
    
    def set_resources(self, service_name: str, replicas: float):
        service = self.client.services.get(service_name)
        service.scale(int(replicas))
