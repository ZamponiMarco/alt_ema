from abc import ABC, abstractmethod
import docker
from docker.errors import APIError, NotFound
import os
import sys
from libs.simulator.docker.app.app_component import AppComponent
from libs.simulator.docker.app.app_info import AppInfo
from libs.simulator.docker.app.container_app_component import ContainerAppComponent
from libs.simulator.docker.app.service_app_component import ServiceAppComponent
from libs.simulator.docker.controller.docker_controller import ContainerController, DockerController, ServiceController
from libs.simulator.docker.monitoring.container_metrics import ContainerMetrics
from libs.simulator.docker.monitoring.prometheus_metrics import PrometheusMetrics
from libs.simulator.docker.monitoring.service_metrics import ServiceMetrics

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROMETHEUS_CONFIG_FILE = os.path.join(CURRENT_DIR, "..", "..", "..", "resources", "prometheus.yml")
TRAEFIK_MIDDLEWARES_FILE = os.path.join(CURRENT_DIR, "..", "..", "..", "resources", "middlewares.yml")

class DockerNetworkInterface(ABC):
    def __init__(self, app_info: AppInfo):
        self.client = docker.from_env()
        
        self.containers = []  # To track started containers
        self.services = []    # To track started services
        self.network = None
        
        if not os.path.exists(PROMETHEUS_CONFIG_FILE):
            print(f"Prometheus configuration file not found at {PROMETHEUS_CONFIG_FILE}")
            
        self.app_info = app_info
        
        self.service_builder: AppComponent = None
        self.monitoring: PrometheusMetrics = None
        self.controller: DockerController = None

    def create_network(self, network_name="traefik"):
        try:
            self.network = self.client.networks.get(network_name)
            print(f"Docker network '{network_name}' already exists.")
        except NotFound:
            self.network = self.client.networks.create(network_name, driver="overlay", attachable=True)
            print(f"Created Docker network '{network_name}'.")

    def start_container(self, **kwargs):
        try:
            container = self.client.containers.run(detach=True, **kwargs)
            print(f"Started container '{kwargs.get('name')}'.")
            self.containers.append(container)
            return container
        except APIError as e:
            print(f"Error starting container '{kwargs.get('name')}': {e}")
            return None

    def start_service(self, **kwargs):
        try:
            service = self.client.services.create(**kwargs)
            print(f"Started service '{kwargs.get('name')}'.")
            self.services.append(service)
            return service
        except APIError as e:
            print(f"Error starting service '{kwargs.get('name')}': {e}")
            return None

    def shutdown(self):
        for service in self.services:
            try:
                print(f"Removing service '{service.name}'...")
                service.remove()
                print(f"Service '{service.name}' removed.")
            except Exception as e:
                print(f"Error removing service '{service.name}': {e}")  
        for container in self.containers:
            try:
                print(f"Stopping container '{container.name}'...")
                container.stop()
                container.remove()
                print(f"Container '{container.name}' stopped and removed.")
            except Exception as e:
                print(f"Error stopping/removing container '{container.name}': {e}")   
        if self.network:
            try:
                print("Removing Docker network 'traefik'...")
                self.network.remove()
                print("Docker network 'traefik' removed.")
            except Exception as e:
                print(f"Error removing Docker network: {e}")

    def run(self):
        self.create_network("traefik")
        traefik_cmd = [
            "--api.insecure=true",
            "--log.level=INFO",
            "--accesslog=true",
            "--providers.file.directory=/etc/traefik/",
            "--providers.docker=true",
            "--providers.docker.exposedbydefault=false",
            "--providers.swarm.refreshSeconds=5",
            "--entryPoints.web.address=:80",
            "--metrics.prometheus=true",
            "--metrics.prometheus.addEntryPointsLabels=false",
        ]
        traefik_container = self.start_container(
            image="traefik",
            name="traefik",
            command=traefik_cmd,
            ports={"80/tcp": 80, "8080/tcp": 8080},
            volumes={
                "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "ro"},
                TRAEFIK_MIDDLEWARES_FILE: {"bind": "/etc/traefik/middlewares.yml", "mode": "ro"}, 
            },
            network="traefik"
        )
        if traefik_container is None:
            raise Exception("Traefik service failed to start.")

        prometheus_container = self.start_container(
            image="prom/prometheus:latest",
            name="prometheus",
            command=["--config.file=/etc/prometheus/prometheus.yml"],
            ports={"9090/tcp": 9090},
            volumes={PROMETHEUS_CONFIG_FILE: {"bind": "/etc/prometheus/prometheus.yml", "mode": "ro"}},
            network="traefik"
        )
        if prometheus_container is None:
            raise Exception("Prometheus service failed to start.")

        cadvisor_container = self.start_container(
            image="gcr.io/cadvisor/cadvisor:v0.52.0",
            name="cadvisor",
            hostname="cadvisor",
            platform="linux/arm",
            command=["-port=8081"],
            volumes={
                "/": {"bind": "/rootfs", "mode": "ro"},
                "/var/run": {"bind": "/var/run", "mode": "ro"},
                "/sys": {"bind": "/sys", "mode": "ro"},
                "/var/lib/docker/": {"bind": "/var/lib/docker", "mode": "ro"},
                "/dev/disk/": {"bind": "/dev/disk", "mode": "ro"}
            },
            ports={"8081/tcp": 8081},
            network="traefik",
        )
        if cadvisor_container is None:
            raise Exception("cAdvisor service failed to start.")
        
        for container_info in self.app_info.requirements:
            ContainerAppComponent(self, container_info).start()
        
        for container_info in self.app_info.services:
            self.service_builder(self, container_info).start()


class ContainerDockerNetwork(DockerNetworkInterface):
    def __init__(self, app_info: AppInfo):
        super().__init__(app_info)
        self.service_builder = ContainerAppComponent
        self.monitoring = ContainerMetrics()
        self.controller = ContainerController(self.client)

class ServiceDockerNetwork(DockerNetworkInterface):
    def __init__(self, app_info: AppInfo):
        super().__init__(app_info)
        self.service_builder = ServiceAppComponent
        self.monitoring = ServiceMetrics()
        self.controller = ServiceController(self.client)