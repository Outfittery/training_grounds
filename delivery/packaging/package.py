import re
from typing import *
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

from yo_fluq_ds import FileIO, Query

from .packaging_dto import PackagingTask, PackageInfo
from pathlib import Path

from .entry_point import EntryPoint
from ..._common import Loc



def _full_module_name(name, version):
    return (name + '__' + version).replace('.', '_').replace('-', '_')


def make_package(
        task: PackagingTask,
        dst_location: Optional[Union[Path, str]] = None) -> PackageInfo:
    """
    Creates the package out of the :class:``PackagingTask``, and returns :class:``PackagingInfo``` describing this package
    """
    if dst_location is None:
        dst_location = Loc.temp_path.joinpath('release/package')
    elif isinstance(dst_location, str):
        dst_location = Path(dst_location)
    elif not isinstance(dst_location, Path):
        raise ValueError(f'dst_location was {dst_location}, while str or Path is expected')
    if not os.path.isdir(dst_location):
        os.makedirs(dst_location, exist_ok=True)

    root = Loc.tg_path  # type:Path
    release = Loc.temp_path.joinpath('release/package_tmp')  # type:Path
    try:
        shutil.rmtree(release.__str__())
    except:
        pass
    os.makedirs(release.__str__())

    full_module_name = _full_module_name(task.name, task.version)
    lib = release.joinpath(full_module_name)

    shutil.copytree(root.__str__(),
                    lib.joinpath(Loc.tg_name).__str__())

    resources = lib.joinpath('resources')  # type: Path
    os.makedirs(resources.__str__())

    props = dict(
        module_name=task.name,
        version=task.version,
        full_module_name=full_module_name,
        dependencies=','.join(f"'{z}'" for dep_list in task.dependencies for z in dep_list.dependencies),
        tg_name=Loc.tg_name,
        full_tg_name=full_module_name + '.' + Loc.tg_name,
    )

    for key, value in task.payload.items():
        FileIO.write_pickle(value, resources.joinpath(key))

    FileIO.write_text(_MANIFEST_TEMPLATE.format(**props), release.joinpath('MANIFEST.in'))
    FileIO.write_text(_SETUP_TEMPLATE.format(**props), release.joinpath('setup.py'))
    FileIO.write_json(props, release.joinpath('properties.json'))

    FileIO.write_text(_INIT_TEMPLATE.format(**props), lib.joinpath('__init__.py'))

    pwd = os.getcwd()
    os.chdir(release.__str__())

    subprocess.call([sys.executable, 'setup.py', 'sdist'])

    os.chdir(pwd)

    file = Query.folder(release.joinpath('dist')).single()

    dst_location = dst_location.joinpath(f'{full_module_name}-{task.version}.tar.gz')

    shutil.copy(file.__str__(), dst_location.__str__())
    shutil.rmtree(release.__str__())
    return PackageInfo(
        task,
        full_module_name,
        dst_location
    )


def get_loader_from_installed_package(name: str) -> EntryPoint:
    """
    By a given name, assuming the name belongs to the training ground's based package, returns the :class:``EntryPoint`` from this package
    """
    module = importlib.import_module(name)
    my_element = getattr(module, 'Entry')
    return my_element


def _get_module_name_and_version(path: Path):
    try:
        file = tarfile.open(path, 'r:gz')
        properties = (Query
                      .en(file.getmembers())
                      .where(lambda z: z.name.endswith('properties.json'))
                      .order_by(lambda z: len(z.name))
                      .first()
                      )
        stream = file.extractfile(properties).read()
        props = json.loads(stream)
        return props['full_module_name'], props['version']
    except:
        module_name, version = re.match('([^-/]+)-(.+)\.tar\.gz$', path.name).groups()
        return module_name, version


def install_package_and_get_loader(filepath: Union[Path, str], try_uninstall_existing=True, silent=False) -> EntryPoint:
    """
    Installs the package from the specified file.
    Args:
        filepath: path to file
        try_uninstall_existing: If True, uninstall existing package of the same name
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    filename = filepath.name

    module_name, version = _get_module_name_and_version(filepath)
    silent_option = ['-q'] if silent else []

    if try_uninstall_existing:
        subprocess.call([sys.executable, '-m', 'pip', 'uninstall', module_name, '-y'] + silent_option)
    subprocess.call([sys.executable, '-m', 'pip', 'install', filepath] + silent_option)
    return get_loader_from_installed_package(module_name)


_INIT_TEMPLATE = '''
from .tg.common.delivery.packaging.entry_point import EntryPoint
from pathlib import Path

Entry = EntryPoint(
    '{module_name}',
    '{version}',
    '{full_module_name}',
    '{full_tg_name}',
    '{tg_name}',
    Path(__file__).parent.joinpath('resources')
)
'''

_MANIFEST_TEMPLATE = '''
include {full_module_name}/resources/*
recursive-include {full_module_name} *.yml *.json *.py *.rst
include properties.json
'''

_SETUP_TEMPLATE = '''
from setuptools import setup, find_packages


setup(name='{module_name}',
      version='{version}',
      description='The framework for featurization and model training',
      packages=find_packages(),
      install_requires=[
          {dependencies}

      ],
      include_package_data = True,
      zip_safe=False)
'''
