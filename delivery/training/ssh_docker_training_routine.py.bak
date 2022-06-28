from typing import *

import copy
import os
import shutil

from pathlib import Path
from subprocess import call
from yo_fluq_ds import Query

from .architecture import TrainingExecutor, AttachedTrainingExecutor, TrainingRoutineBase, ResultPickleReader, _create_id, _TRAINING_RESULTS_LOCATION
from ..jobs import ssh_docker_job_routine as jobs
from .. import packaging as pkg
from ...ml.training_core import AbstractTrainingTask



_ENTRY_FILE_TEMPLATE = '''
import {module}.{tg_name}.common.delivery.training.ssh_docker_execution as feat
from {module} import Entry

feat.execute(Entry)
'''


def _build_container(
        task: AbstractTrainingTask,
        name: str,
        version: str,
        tag: str,
        dependencies: Optional[List[pkg.DependenciesList]] = None
):
    task = pkg.ContaineringTask(
        pkg.PackagingTask(
            name,
            version,
            dict(job=task),
            dependencies
        ),
        'run.py',
        _ENTRY_FILE_TEMPLATE,
        jobs.DOCKERFILE_TEMPLATE,
        name,
        tag,
    )
    pkg.make_container(task)

class SSHDockerTrainingRoutine(TrainingRoutineBase):
    def __init__(self,
                 local_dataset_storage: Path,
                 common_task_name: str,
                 local_home_folder: Path,
                 remote_home_folder: Path,
                 repository: str,
                 remote_host_address: str,
                 remote_host_user: str,
                 additional_dependencies: List[pkg.DependenciesList],
                 docker_options: jobs.DockerOptions = None
                 ):
        super(SSHDockerTrainingRoutine, self).__init__(local_dataset_storage)
        self._repository = repository
        self._common_task_name = common_task_name
        self._repository = repository
        self._remote_host_address = remote_host_address
        self._remote_host_user = remote_host_user
        self.additional_dependencies = additional_dependencies

        self.local = LocalDockerTrainingExecutor(self, local_home_folder)
        self.remote = RemoteDockerTrainingExecutor(self, remote_home_folder)
        self.attached = AttachedTrainingExecutor(self, )
        self.docker_options = docker_options





class DockerTrainingExecutor(TrainingExecutor):
    def __init__(self, routine: SSHDockerTrainingRoutine, home_folder: Path):
        self.routine = routine
        self.home_folder = home_folder

    def execute(self, task, dataset_version, wait=True):
        id = _create_id(task.info.get("name", ""))
        task_output = os.path.join(str(self.home_folder), 'output', id)
        task.info['run_on_dataset'] = dataset_version
        self._run_internal(task, id, task_output, dataset_version, wait)
        return id

    def _run_internal(self, task, id, task_output_folder, dataset_version, wait):
        raise NotImplementedError()


    def _download_result_file(self, id: str, location: Path):
        raise NotImplementedError()

    def get_result(self, job_id) -> ResultPickleReader:
        """
        Downloads model artifacts for the given job_id
        """
        location = _TRAINING_RESULTS_LOCATION
        os.makedirs(location, exist_ok=True)
        location = Path(location)
        tar_file = location/f'{job_id}.tar.gz'
        self._download_result_file(job_id, tar_file)
        return ResultPickleReader.from_tar_gz_file(tar_file, job_id, location)



class LocalDockerTrainingExecutor(DockerTrainingExecutor):
    def __init__(self, training_info, home_folder):
        super(LocalDockerTrainingExecutor, self).__init__(training_info, home_folder)

    def upload_dataset(self, dataset_version: str):
        pass

    def _run_internal(self, task, id, task_output_folder, dataset_version, wait):
        os.makedirs(task_output_folder)
        options = copy.deepcopy(self.routine.docker_options)
        if options is None:
            options = jobs.DockerOptions()
        if options.mount_volumes is None:
            options.mount_volumes = {}
        options.mount_volumes['/home/tg'] = self.home_folder
        options.mount_volumes['/home/data'] = self.routine.local_dataset_storage/dataset_version
        options.mount_volumes['/home/output'] = task_output_folder

        tag = self.routine._common_task_name+'-'+id
        _build_container(task, self.routine._common_task_name, id, tag, self.routine.additional_dependencies)

        call_args = jobs.get_docker_run_cmd(f'{self.routine._common_task_name}:{tag}', '', options)
        call(call_args)

    def upload_shared_data(self, source_path: Path, dst_key: str):
        dst_path = self.home_folder/dst_key
        if source_path.is_file():
            os.remove(str(dst_path))
            shutil.copy(str(source_path), str(dst_path))
        else:
            shutil.rmtree(dst_path, ignore_errors=True)
            shutil.copytree(str(source_path), str(self.home_folder/dst_key))

    def _download_result_file(self, id: str, location: Path):
        shutil.copy(
            str(self.home_folder/'output'/id/'result.tar.gz'),
            str(location)
        )




class _SSHAccessor:
    def __init__(self, address: str, username: str):
        self.address = address
        self.username = username

    def _make_remote_dir(self, host, username, folder):
        command = jobs.get_ssh(self.address, self.username) + ['mkdir', '-p', str(folder)]
        if call(command) != 0:
            raise ValueError(f"Command failed:{command}")


    def download_file(self, remote_path, local_path):
        folder = Path(local_path).parent
        os.makedirs(folder,exist_ok=True)
        command = [
            'scp',
            f'{self.username}@{self.address}:{remote_path}',
            local_path
        ]
        call(command)

    def make_remote_dir(self, folder):
        command = jobs.get_ssh(self.address, self.username) + ['mkdir', '-p', str(folder)]
        if call(command)!=0:
            raise ValueError(f"Command failed:{command}")

    def upload_file(self, local_path, remote_path):
        folder = Path(remote_path).parent
        self.make_remote_dir(folder)
        command = [
            'scp',
            local_path,
            f'{self.username}@{self.address}:{remote_path}'
        ]
        call(command)

    def upload_folder(self, local_path, remote_path):
        for file in Query.folder(local_path,'**/*').where(lambda z: z.is_file()):
            inner = str(file)[len(str(local_path)):]
            if inner.startswith('/'):
                inner = inner[1:]
            self.upload_file(os.path.join(local_path, inner), os.path.join(remote_path, inner))




class RemoteDockerTrainingExecutor(DockerTrainingExecutor):
    def __init__(self, training_info, home_folder):
        super(RemoteDockerTrainingExecutor, self).__init__(training_info, home_folder)

    def _run_internal(self,  task, id, task_output_folder, dataset_version, wait):
        _SSHAccessor(self.routine._remote_host_address, self.routine._remote_host_user).make_remote_dir(task_output_folder)

        options = copy.deepcopy(self.routine.docker_options)
        if options.mount_volumes is None:
            options.mount_volumes = {}
        options.mount_volumes['/home/tg'] = self.home_folder
        options.mount_volumes['/home/data'] = self.home_folder/'training_data'/dataset_version
        options.mount_volumes['/home/output'] = task_output_folder

        tag = self.routine._common_task_name + '-' + id
        _build_container(task, self.routine._common_task_name, id, tag, self.routine.additional_dependencies)
        remote_image_name = self.routine._repository+":"+tag
        pkg.push_container_to_quay(self.routine._common_task_name, tag, remote_image_name)
        ssh = jobs.get_ssh(self.routine._remote_host_address, self.routine._remote_host_user)

        call(ssh + ['docker', 'pull', remote_image_name])

        options.wait_for_stop = wait
        docker_cmd = jobs.get_docker_run_cmd(remote_image_name, '"', options)
        print(docker_cmd)
        call(ssh + docker_cmd)

    def upload_shared_data(self, source_path: Path, dst_key: str):
        ac = _SSHAccessor(self.routine._remote_host_address, self.routine._remote_host_user)
        if source_path.is_file():
            ac.upload_file(source_path, self.home_folder/dst_key)
        else:
            ac.upload_folder(source_path, self.home_folder/dst_key)

    def kill(self, id):
        command = (
                jobs.get_ssh(self.routine._remote_host_address, self.routine._remote_host_user) +
                jobs.kill_by_full_image_name(f'{self.routine._repository}:{self.routine._common_task_name}-{id}')
        )
        call(command)

    def get_logs(self, id):
        ssh = jobs.get_ssh(self.routine._remote_host_address, self.routine._remote_host_user)
        return jobs.get_logs_by_full_image_name(f'{self.routine._repository}:{self.routine._common_task_name}-{id}', ssh, False)


    def _download_result_file(self, id: str, location: Path):
        ac = _SSHAccessor(self.routine._remote_host_address, self.routine._remote_host_user)
        ac.download_file(self.home_folder/'output'/id/'result.tar.gz', location)

    def upload_dataset(self, dataset_version: str):
        self.upload_shared_data(
            self.routine.local_dataset_storage/dataset_version,
            f'training_data/{dataset_version}'
        )

