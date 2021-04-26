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


def push_contaner_to_aws(
        image_name: str,
        image_tag: str,
        region: str = '',
        registry: str = ''):
    """
    Pushes the previously build local docker container to AWS ECR, from where it can be later used by Sagemaker
    """
    p1 = subprocess.Popen(['aws', 'ecr', 'get-login-password', '--region', region], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['docker', 'login', '-u', 'AWS', '--password-stdin', registry.split('/')[0]], stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.communicate()

    remote_name = f'{registry}:{image_tag}'

    call([
        'docker',
        'tag',
        image_name + ":" + image_tag,
        remote_name
    ])

    call([
        'docker',
        'push',
        remote_name
    ])


def push_container_to_quay(name: str, tag: str, remote_image_name_and_tag):
    call([
        'docker',
        'tag',
        f'{name}:{tag}',
        remote_image_name_and_tag
    ])
    call([
        'docker',
        'push',
        remote_image_name_and_tag
    ])


_run_py_template = '''
from {module}.{tg_name}.{entry_module} import {entry_method} as entry_method
from {module} import Entry

entry_method(Entry)


'''
