from typing import *
from ..delivery import Container, EntryPoint
import os
import subprocess


class SSHDockerOptions:
    def __init__(self,
                 env_variables_to_propagate: Optional[Iterable[str]] = None,
                 wait_for_stop: bool = True,
                 cpu_limit: Optional[float] = None,
                 memory_limit_in_gygabytes: Optional[float] = None
                 ):
        self.env_variables_to_propagate = env_variables_to_propagate
        self.cpu_limit = cpu_limit
        self.wait_for_stop = wait_for_stop
        self.memory_limit_in_gygabytes = memory_limit_in_gygabytes


class SSHDockerConfig:
    def __init__(self,
                 container: Container,
                 options: SSHDockerOptions,
                 username: Optional[str] = None,
                 host: Optional[str] = None
                 ):
        self.container = container
        self.options = options
        self.username = username
        self.host = host

    def get_docker_run_cmd(self, image_to_call, environment_quotation):
        envs = []
        if self.options.env_variables_to_propagate is not None:
            for var in self.options.env_variables_to_propagate:
                if var in os.environ:
                    envs.append('--env')
                    envs.append(f'{environment_quotation}{var}={os.environ[var]}{environment_quotation}')

        additional = []
        if not self.options.wait_for_stop:
            additional.append('--detach')
        if self.options.cpu_limit is not None:
            additional.append(f'--cpus={self.options.cpu_limit}')
        if self.options.memory_limit_in_gygabytes is not None:
            additional.append(f'--memory={self.options.memory_limit_in_gygabytes}g')

        return [
            'docker',
            'run',
            *envs,
            *additional,
            image_to_call
        ]

    def get_ssh(self):
        if self.username is None or self.host is None:
            raise ValueError('SSH command was requested, but `username` or `host` were not provided')
        ssh = [
            'ssh',
            f'{self.username}@{self.host}'
        ]
        return ssh

    @staticmethod
    def get_logs_by_full_image_name(full_image_name, ssh_prefix) -> Tuple[str, str]:
        ps_call = ['docker', 'ps', '-a', '-q', '--filter', 'ancestor=' + full_image_name, '--format={{.ID}}']
        container_id = subprocess.check_output(ssh_prefix + ps_call)
        container_id = container_id.decode('utf-8').strip()
        p = subprocess.Popen(ssh_prefix + ["docker", "logs", container_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        return out.decode('utf-8'), err.decode('utf-8')


class SSHAttachedExecutor:
    def __init__(self, config: SSHDockerConfig):
        self.config = config

    def execute(self):
        entry_point = EntryPoint(self.config.container.name, self.config.container.tag, None, self.config.container.entry_point)
        entry_point.run()


class SSHLocalExecutor:
    def __init__(self, config: SSHDockerConfig):
        self.config = config

    def get_full_image_name(self):
        return f'{self.config.container.name}:{self.config.container.tag}'

    def execute(self):
        self.config.container.build()
        full_image_name = self.get_full_image_name()
        call_args = self.config.get_docker_run_cmd(full_image_name, '')
        subprocess.call(call_args)

    def get_logs(self) -> Tuple[str, str]:
        full_image_name = self.get_full_image_name()
        result = SSHDockerConfig.get_logs_by_full_image_name(full_image_name, [])
        return result


class SSHRemoteExecutor():
    def __init__(self, config: SSHDockerConfig):
        self.config = config

    def call(self, cmd):
        #print(' '.join(cmd))
        subprocess.call(cmd)

    def execute(self):
        # creating container and pushing it
        self.config.container.build()
        self.config.container.push()

        self.call(self.config.get_ssh() + list(self.config.container.pusher.get_auth_command()))


        remote_name = self.config.container.get_remote_name()
        self.call(self.config.get_ssh() + ['docker', 'pull', remote_name])

        run_command = self.config.get_docker_run_cmd(remote_name, '"')
        self.call(self.config.get_ssh() + run_command)

    def get_logs(self):
        result = SSHDockerConfig.get_logs_by_full_image_name(
            self.config.container.get_remote_name(),
            self.config.get_ssh()
        )
        return result







