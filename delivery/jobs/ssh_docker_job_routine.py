from typing import *

import subprocess
import os

from ..packaging.packaging_dto import ContaineringTask, PackagingTask
from .architecture import DeliverableJob, JobExecutor, DockerOptions
from ..packaging import make_container, push_container_to_quay




ENTRY_FILE_TEMPLATE = '''
import {module}.{tg_name}.common.delivery.jobs.ssh_docker_job_execution as feat
from {module} import Entry

feat.execute_featurization_job(Entry)
'''

DOCKERFILE_TEMPLATE = '''FROM python:3.7

{install_libraries}

COPY . /featurization

WORKDIR /featurization

COPY {package_filename} package.tar.gz

RUN pip install package.tar.gz

CMD ["python3","/featurization/run.py"]
'''


def build_container(job: DeliverableJob, name: str, version: str, tag: str, entry_file_template=ENTRY_FILE_TEMPLATE):
    task = ContaineringTask(
        PackagingTask(
            name,
            version,
            dict(job=job)
        ),
        'run.py',
        entry_file_template,
        DOCKERFILE_TEMPLATE,
        name,
        tag
    )
    make_container(task)


def get_docker_run_cmd(image_to_call, environment_quotation, options: DockerOptions):
    envs = []
    if options.propagate_environmental_variables is not None:
        for var in options.propagate_environmental_variables:
            envs.append('--env')
            envs.append(f'{environment_quotation}{var}={os.environ[var]}{environment_quotation}')

    mounts = []
    if options.mount_volumes is not None:
        for mount_to, mount_from in options.mount_volumes.items():
            mounts.append('--mount')
            mounts.append(f"type=bind,source={mount_from},target={mount_to}")

    additional = []
    if not options.wait_for_stop:
        additional.append('--detach')
    if options.cpu_limit is not None:
        additional.append(f'--cpus={options.cpu_limit}')
    if options.memory_limit_in_gygabytes is not None:
        additional.append(f'--memory={options.memory_limit_in_gygabytes}g')

    return [
        'docker',
        'run',
        *envs,
        *mounts,
        *additional,
        image_to_call
    ]


def get_ssh(host, username):
    ssh = [
        'ssh',
        f'{username}@{host}'
    ]
    return ssh


def kill_by_full_image_name(full_image_name):
    return ["docker",
            "rm",
            '$(docker',
            'stop $(docker ps -a -q --filter ancestor=' + full_image_name + '  --format="{{.ID}}"))'
            ]


def get_logs_by_full_image_name(full_image_name, ssh_prefix, truncate_quotes) -> Tuple[str, str]:
    ps_call = ['docker', 'ps', '-a', '-q', '--filter', 'ancestor=' + full_image_name, '--format="{{.ID}}"']
    container_id = subprocess.check_output(ssh_prefix + ps_call)
    if truncate_quotes:
        container_id = container_id[1:-2]
    else:
        container_id = container_id[:-1]
    p = subprocess.Popen(ssh_prefix + ["docker", "logs", container_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out.decode('utf-8'), err.decode('utf-8')


class SSHDockerJobRoutine:
    def __init__(self,
                 job: DeliverableJob,
                 repository: str,
                 remote_host_address: str,
                 remote_host_user: str,
                 options: DockerOptions
                 ):
        self._job = job

        self._name, self._version = job.get_name_and_version()
        self._tag = self._name + '-' + self._version
        self._remote_image_name = f'{repository}:{self._tag}'

        self.remote_host_address = remote_host_address
        self.remote_host_user = remote_host_user

        self._options = options or DockerOptions()

        self.attached = AttachedJobExecutor(self)
        self.local = LocalJobExecutor(self)
        self.remote = RemoteJobExecutor(self)


class RemoteJobExecutor(JobExecutor):
    def __init__(self, routine: SSHDockerJobRoutine):
        self.routine = routine

    def execute(self):
        build_container(self.routine._job, self.routine._name, self.routine._version, self.routine._tag)
        push_container_to_quay(self.routine._name, self.routine._tag, self.routine._remote_image_name)
        ssh = get_ssh(self.routine.remote_host_address, self.routine.remote_host_user)

        subprocess.call(ssh + ['docker', 'pull', self.routine._remote_image_name])
        subprocess.call(ssh + get_docker_run_cmd(self.routine._remote_image_name, '"', self.routine._options))

    def kill(self):
        ssh = get_ssh(self.routine.remote_host_address, self.routine.remote_host_user)
        subprocess.call(ssh + kill_by_full_image_name(self.routine._remote_image_name))

    def get_logs(self) -> Tuple[str, str]:
        ssh = get_ssh(self.routine.remote_host_address, self.routine.remote_host_user)
        result = get_logs_by_full_image_name(self.routine._remote_image_name, ssh, False)
        return result


class LocalJobExecutor(JobExecutor):
    def __init__(self, routine: SSHDockerJobRoutine):
        self.routine = routine

    def execute(self):
        build_container(self.routine._job, self.routine._name, self.routine._version, self.routine._tag)
        call_args = get_docker_run_cmd(f'{self.routine._name}:{self.routine._tag}', '', self.routine._options)
        subprocess.call(call_args)

    def get_logs(self) -> Tuple[str, str]:
        result = get_logs_by_full_image_name(f'{self.routine._name}:{self.routine._tag}', [], True)
        return result


class AttachedJobExecutor(JobExecutor):
    def __init__(self, routine: SSHDockerJobRoutine):
        self.routine = routine

    def execute(self):
        self.routine._job.run()
