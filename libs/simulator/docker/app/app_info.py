class ContainerInfo():
    
    def __init__(self, name: str, image: str, env: dict, path: str, volumes: dict = None):
        self.name = name
        self.image = image
        self.env = env
        self.path = path
        self.volumes = volumes if volumes else {}

class AppInfo():
    
    def __init__(self, requirements: list[ContainerInfo], services: list[ContainerInfo]):
        self.requirements = requirements
        self.services = services