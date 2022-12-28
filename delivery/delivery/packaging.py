from typing import *

import os
import shutil
import subprocess
import sys

from yo_fluq_ds import FileIO, Query
from pathlib import Path

from .templates import PACKAGE_INIT_TEMPLATE, PACKAGE_SETUP_PY_TEMPLATE, PACKAGE_MANFEST_TEMPLATE
from ..._common import Loc

class DependencyList:
    def __init__(self, name, dependencies: Iterable[str]):
        self.name = name
        self.dependencies = tuple(dependencies)

    def __repr__(self):
        return {self.name: list(self.dependencies)}.__repr__()


def get_minimal_dependencies():
    return (DependencyList('min', ['boto3', 'yo_fluq_ds', 'simplejson']), )


def _full_module_name(name, version):
    return (name + '__' + version).replace('.', '_').replace('-', '_')


class Packaging:
    def __init__(self, name: str, version: str, payload: Dict[str, Any]):
        self.name = name
        self.version = version
        self.payload = payload


        self.dependencies = get_minimal_dependencies()
        self.human_readable_module_name = False

        self.callback_before_calling_template = None # type: Optional[Callable]
        self.manifest_template = PACKAGE_MANFEST_TEMPLATE
        self.setup_py_template = PACKAGE_SETUP_PY_TEMPLATE
        self.init_py_template = PACKAGE_INIT_TEMPLATE

        self.package_location = None #type: Optional[Path]
        self.package_properties = None #type: Optional[Dict]
        self.package_module_name = None #type: Optional[str]

        self.silent = False


    @staticmethod
    def get_job_name_and_version(job):
        if hasattr(job, 'get_name'):
            return job.get_name(), '0'
        if hasattr(job, 'get_name_and_version'):
            return job.get_name_and_version()
        return type(job).__name__.lower(), '0'


    def make_package(
            self,
            dst_location: Optional[Union[Path, str]] = None,
        ):
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
        release = Loc.temp_path/'release/package_tmp'  # type:Path
        try:
            shutil.rmtree(release.__str__())
        except:
            pass
        os.makedirs(release.__str__())

        if not self.human_readable_module_name:
            full_module_name = _full_module_name(self.name, self.version)
        else:
            full_module_name = self.name

        lib = release / full_module_name

        shutil.copytree(str(root), str(lib/Loc.tg_name))

        resources = lib/'resources'  # type: Path
        os.makedirs(str(resources))

        props = dict(
            name=self.name,
            version=self.version,
            module_name=full_module_name,
            dependencies=','.join(f"'{z}'" for dep_list in self.dependencies for z in dep_list.dependencies),
            original_tg_import_path=Loc.tg_name,
            tg_import_path=full_module_name + '.' + Loc.tg_name,
        )

        if self.callback_before_calling_template:
            self.callback_before_calling_template(properties = props, job = self)

        for key, value in self.payload.items():
            FileIO.write_pickle(value, resources.joinpath(key))

        FileIO.write_text(self.manifest_template.format(**props), release.joinpath('MANIFEST.in'))
        FileIO.write_text(self.setup_py_template.format(**props), release.joinpath('setup.py'))
        FileIO.write_text(self.init_py_template.format(**props), lib.joinpath('__init__.py'))
        FileIO.write_json(props, release.joinpath('properties.json'))

        pwd = os.getcwd()
        os.chdir(release.__str__())

        args = [sys.executable, 'setup.py']
        if self.silent:
            args.append('-q')
        args.append('sdist')
        subprocess.call(args)

        os.chdir(pwd)

        file = Query.folder(release.joinpath('dist')).single()

        dst_location = dst_location.joinpath(f'{full_module_name}-{self.version}.tar.gz')

        shutil.copy(file.__str__(), dst_location.__str__())
        shutil.rmtree(release.__str__())

        self.package_location = dst_location
        self.package_properties = props
        self.package_module_name = full_module_name
        return self

