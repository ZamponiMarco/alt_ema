from libs.simulator.docker.app.app_component import AppComponent
from libs.simulator.docker.app.app_info import AppInfo


class ContainerAppComponent(AppComponent):
    def start(self):
        container = self.docker_network.start_container(
            name=self.container_info.name,
            image=self.container_info.image,
            environment=self.container_info.env,
            labels={
                "traefik.enable": "true",
                f"traefik.http.routers.{self.container_info.name}.entrypoints": "web",
                f"traefik.http.routers.{self.container_info.name}.rule": f"PathPrefix(`/{self.container_info.path}`)",
                f"traefik.http.routers.{self.container_info.name}.middlewares": f"service-strip@file",
            },
            volumes=self.container_info.volumes,
            cpu_quota=100000,
            cpu_period=100000,
            network="traefik"
        )
        if container is None:
            raise Exception(f"Application container {self.container_info.name} failed to start.")
        return container