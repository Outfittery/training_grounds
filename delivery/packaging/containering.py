from typing import *
from .packaging_dto import ContaineringTask
from subprocess import call

from .package import make_package
from yo_fluq_ds import FileIO
from pathlib import Path
import shutil
import subprocess
import os

from ..._common.locations import Loc



def make_container(task: ContaineringTask):
    """
    Creates a package of the current training grounds, and then uses the package
    to create a local container that trains the provided model
    """
    release = Loc.temp_path.joinpath('release/container')  # type:Path
    os.makedirs(release.__str__(), exist_ok=True)

    packaging_info = make_package(task.packaging_task, release)

    install_libraries = ''
    for dep_list in task.packaging_task.dependencies:
        install_libraries+='RUN pip install ' + ' '.join(dep_list.dependencies) + "\n\n"

    props = dict(
        module=packaging_info.module_name,
        tg_name=Loc.tg_name,
        install_libraries = install_libraries,
        package_filename=packaging_info.path.name
    )


    entry_file = task.entry_file_template.format(**props)
    FileIO.write_text(entry_file, release.joinpath(task.entry_file_name))

    docker_file = task.dockerfile_template.format(**props)

    FileIO.write_text(docker_file, release.joinpath('Dockerfile'))

    call([
        'docker',
        'build',
        '-t',
        task.image_name + ":" + task.image_tag,
        release.__str__()
    ])

    shutil.rmtree(release)


_run_py_template = '''
from {module}.{tg_name}.{entry_module} import {entry_method} as entry_method
from {module} import Entry

entry_method(Entry)


'''

class ContainerPusher:
    def get_auth_command(self):
        raise  NotImplementedError()

    def get_remote_name(self, image_name: str, image_tag: str):
        raise NotImplementedError()

    def push(self, image_name: str, image_tag: str):
        raise NotImplementedError()


class ContainerHandler:
    def get_image_name(self) -> str:
        raise NotImplementedError()

    def get_tag(self) -> str:
        raise NotImplementedError()

    def get_auth_command(self) -> Optional[List[str]]:
        raise  NotImplementedError()

    def get_remote_name(self) -> str:
        raise NotImplementedError()

    def push(self) -> None:
        raise NotImplementedError()

    class Factory:
        def create_handler(self, name: str, version: str) -> 'ContainerHandler':
            raise NotImplementedError()


class FakeContainerHandler(ContainerHandler):
    def __init__(self, name, version):
        self.name = name
        self.version = version

    def get_image_name(self) -> str:
        return self.name

    def get_tag(self) -> str:
        return self.version

    class Factory(ContainerHandler.Factory):
        def create_handler(self, name: str, version: str) -> 'ContainerHandler':
            return FakeContainerHandler(name, version)