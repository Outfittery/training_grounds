from typing import *
import sys
import subprocess
from .entry_point import EntryPoint
import importlib
from yo_fluq_ds import Query
from pathlib import Path
import tarfile
import json



def get_loader_from_installed_package(name: str) -> EntryPoint:
    """
    By a given name, assuming the name belongs to the training ground's based package, returns the :class:``EntryPoint`` from this package
    """
    module = importlib.import_module(name)
    my_element = getattr(module, 'Entry')
    return my_element


def _read_file(file, fname_ending):
    fname = (Query
             .en(file.getmembers())
             .where(lambda z: z.name.endswith(fname_ending))
             .order_by(lambda z: len(z.name))
             .first())
    content = file.extractfile(fname).read().decode('utf-8')
    return content


def _get_old_module_to_remove(path: Path):
    with tarfile.open(path, 'r:gz') as file:
        egg_info = _read_file(file, '.egg-info/PKG-INFO')
        module_to_remove = Query.en(egg_info.split('\n')).select(lambda z: z.split(': ')).where(lambda z: z[0] == 'Name').select(lambda z: z[1]).single()
        return module_to_remove.strip()


def _get_module_to_import(path: Path):
    with tarfile.open(path, 'r:gz') as file:
        props = json.loads(_read_file(file, 'properties.json'))
        return props['module_name']


def uninstall_old_version(filepath: Union[Path, str], silent=False):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    module_to_remove = _get_old_module_to_remove(filepath)

    silent_option = ['-q'] if silent else []
    subprocess.call([sys.executable, '-m', 'pip', 'uninstall', module_to_remove, '-y'] + silent_option)


def install_package_and_get_loader(
        filepath: Union[Path, str],
        try_uninstall_existing=True,
        silent=False) -> EntryPoint:
    """
    Installs the package from the specified file.
    Args:
        filepath: path to file
        try_uninstall_existing: If True, uninstall existing package of the same name
        silent: runs pip with -q option, supressing output
    """
    if try_uninstall_existing:
        uninstall_old_version(filepath)

    if isinstance(filepath, str):
        filepath = Path(filepath)
    filename = filepath.name
    module_to_import = _get_module_to_import(filepath)
    silent_option = ['-q'] if silent else []
    subprocess.call([sys.executable, '-m', 'pip', 'install', filepath] + silent_option)
    return get_loader_from_installed_package(module_to_import)

