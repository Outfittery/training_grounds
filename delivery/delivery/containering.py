from typing import *

import shutil
import subprocess
import os

from subprocess import call
from yo_fluq_ds import FileIO
from pathlib import Path

from ..._common.locations import Loc
from .packaging import Packaging, DependencyList, get_minimal_dependencies
from .container_pusher import ContainerPusher
from .templates import DOCKER_RUN_PY_TEMPLATE, DOCKERFILE_TEMPLATE



class Containering:
    def __init__(self,
                 image_name: str,
                 image_tag: str,
                 dependencies: Iterable[DependencyList],
                 ):
        self.image_name = image_name
        self.image_tag = image_tag
        self.dependencies = get_minimal_dependencies()

        self.python_version = '3.8'
        self.callback_before_calling_template = None  # type: Optional[Callable]
        self.run_file_name = 'run.py'
        self.run_file_template = DOCKER_RUN_PY_TEMPLATE
        self.dockerfile_template = DOCKERFILE_TEMPLATE

        self.pusher = None #type: Optional[ContainerPusher]
        self.silent = False

    @staticmethod
    def from_packaging(packaging: Packaging):
        return Containering(packaging.name.lower(), packaging.version.lower(), packaging.dependencies)

    def make_container(self, packaging: Packaging):
        install_libraries = ''
        for dep_list in self.dependencies:
            install_libraries += 'RUN pip install ' + ' '.join(dep_list.dependencies) + "\n\n"

        release = Loc.temp_path/'release/container'
        os.makedirs(release.__str__(), exist_ok=True)
        shutil.copy(packaging.package_location, release/packaging.package_location.name)

        props = dict(
            module=packaging.package_module_name,
            install_libraries=install_libraries,
            package_filename=packaging.package_location.name,
            python_version = self.python_version,
            run_file_name = self.run_file_name
        )

        if self.callback_before_calling_template is not None:
            self.callback_before_calling_template(properties = props, job = self)

        run_file = self.run_file_template.format(**props)
        FileIO.write_text(run_file, release.joinpath(self.run_file_name))

        docker_file = self.dockerfile_template.format(**props)
        FileIO.write_text(docker_file, release.joinpath('Dockerfile'))

        args = [
            'docker',
            'build'
            ]
        if self.silent:
            args.append('--quiet')
        args.extend([
            '-t',
            self.image_name + ":" + self.image_tag,
            release.__str__()
        ])

        result = call(args)
        if result!= 0:
            raise ValueError('Docker deamon returned non-zero code')
        shutil.rmtree(release)
        return self

    def get_remote_name(self):
        if self.pusher is None:
            raise ValueError("Cannot access remote name if pusher is not set")
        return self.pusher.get_remote_name(self.image_name, self.image_tag)

    def push_container(self):
        self.pusher.push(self.image_name, self.image_tag)

