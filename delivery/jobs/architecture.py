from typing import *



class DeliverableJob:
    def get_name_and_version(self) -> Tuple[str, str]:
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class JobExecutor:
    def execute(self):
        raise NotImplementedError()


class DockerOptions:
    def __init__(self,
                 propagate_environmental_variables: Optional[List[str]] = None,
                 mount_volumes: Optional[Dict[str, str]] = None,
                 wait_for_stop: bool = True,
                 cpu_limit: Optional[float] = None,
                 memory_limit_in_gigabytes: Optional[int] = None,
                 ):
        self.propagate_environmental_variables = propagate_environmental_variables or []
        self.mount_volumes = mount_volumes or {}
        self.wait_for_stop = wait_for_stop
        self.cpu_limit = cpu_limit
        self.memory_limit_in_gygabytes = memory_limit_in_gigabytes
