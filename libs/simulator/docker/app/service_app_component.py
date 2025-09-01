from libs.simulator.docker.app.app_component import AppComponent
from libs.simulator.docker.app.app_info import AppInfo


class ServiceAppComponent(AppComponent):
    def start(self):
        service = self.docker_network.start_service(
            name=self.container_info.name,
            image=self.container_info.image,
            env=self.container_info.env,
            labels={
                "traefik.enable": "true",
                f"traefik.http.routers.{self.container_info.name}.entrypoints": "web",
                f"traefik.http.routers.{self.container_info.name}.rule": f"PathPrefix(`/{self.container_info.path}`)",
                f"traefik.http.services.{self.container_info.name}.loadbalancer.server.port": "80",
                f"traefik.http.routers.{self.container_info.name}.middlewares": f"service-strip@file",
            },
            mode={"Replicated": {"Replicas": 1}},
            resources={"Limits": {"NanoCPUs": 1000000000}},
            networks=["traefik"]
        )
        if service is None:
            raise Exception(f"Application service {self.container_info.name} failed to start.")
        return service