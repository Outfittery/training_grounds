from unittest import TestCase
from tg.common.test_common.test_delivery.test_delivery.class_hmr import _FeaturizerForTestPurposes
from tg.common.delivery.delivery import *
from tg.common.delivery.delivery.package_utilities import _get_old_module_to_remove
import os
from pathlib import Path
from yo_fluq_ds import FileIO
import inspect
import subprocess
import sys

def _replace_file(suffix):
    path = Path(__file__).parent.joinpath('class_hmr.py')
    FileIO.write_text(_TEMPLATE.format(suffix), path)



class PackagingTestCase(TestCase):
    def loader_expectations(self, loader: EntryPoint, full_module_name):
        #Check various expectations about the locations of the source code of the loader
        self.assertTrue(os.path.isdir(loader.resources_location))
        root_folder = Path(loader.resources_location).parent
        self.assertEqual(full_module_name, root_folder.name)
        resources = loader.get_resources()
        obj = loader.load_resource(resources[0])
        def_file = inspect.getfile(type(obj))
        print('##',def_file, root_folder)
        self.assertTrue(def_file.startswith(str(root_folder)))

    def test_human_readable_packaging(self):
        task = Packaging('tg_human_readable', '0.0.0', dict(test=_FeaturizerForTestPurposes('Passed')))
        task.silent = True
        task.human_readable_module_name = True
        task.make_package()
        self.assertEqual('tg_human_readable', task.package_module_name)
        self.assertEqual('tg_human_readable-0.0.0.tar.gz', task.package_location.name)
        loader = install_package_and_get_loader(task.package_location, silent=True)
        self.loader_expectations(loader, 'tg_human_readable')
        uninstall_old_version(task.package_location)
        os.remove(task.package_location.__str__())
        print(task.package_properties)

    def test_nasty_names(self):
        task = Packaging('test-nasty', 'v21', dict(test=_FeaturizerForTestPurposes('Passed')))
        task.silent = True
        task.make_package()
        entry_point = install_package_and_get_loader(task.package_location)

        deps = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')
        module_name = _get_old_module_to_remove(task.package_location) + ' @'
        self.assertIn(module_name, deps)

        uninstall_old_version(task.package_location)
        deps = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')
        self.assertNotIn(module_name, deps)


    def test_packaging(self):
        _replace_file('A')

        task = Packaging('tg_test_package', '0.0.0', dict(test=_FeaturizerForTestPurposes('Passed')))
        task.silent = True
        task.make_package()

        self.assertEqual('tg_test_package__0_0_0', task.package_module_name)
        self.assertEqual('tg_test_package__0_0_0-0.0.0.tar.gz', task.package_location.name)

        # Here we load the package, and the entry point from it
        loader = install_package_and_get_loader(task.package_location, silent=True)
        self.loader_expectations(loader, 'tg_test_package__0_0_0')

        # Checks that we really managed to read the entry from the package
        loaded_entry = loader.load_resource('test')
        self.assertEqual('Passed, package A', loaded_entry())

        # cleaning-up
        uninstall_old_version(task.package_location)
        self.assertFalse(os.path.isdir(loader.resources_location))
        os.remove(task.package_location.__str__())


    def test_hot_module_replacement(self):
        # We build two packages with different source code
        _replace_file('A')
        delA = Packaging('tg_test_hmr', '0.0.1', dict(test=_FeaturizerForTestPurposes('argument A')))
        delA.silent = True
        delA.make_package()

        _replace_file('B')
        delB = Packaging('tg_test_hmr', '0.0.2', dict(test=_FeaturizerForTestPurposes('argument B')))
        delB.silent = True
        delB.make_package()

        # Loading the first package and checking if it works
        loaderA = install_package_and_get_loader(delA.package_location, silent=False)
        self.loader_expectations(loaderA, 'tg_test_hmr__0_0_1')
        self.assertEqual('argument A, package A', loaderA.load_resource('test')())

        # Hot-Replacing the first package with the second one, and checking if it works
        loaderB = install_package_and_get_loader(delB.package_location, silent=False)
        self.loader_expectations(loaderB, 'tg_test_hmr__0_0_2')
        self.assertFalse(os.path.isdir(loaderA.resources_location)) # expect the first version to be uninstalled

        self.assertEqual('argument B, package B', loaderB.load_resource('test')())
        # Cleaning-up
        _replace_file('')
        uninstall_old_version(delB.package_location)
        os.remove(delA.package_location.__str__())
        os.remove(delB.package_location.__str__())

    def test_running_alongside(self):
        # We build two packages with different source code AND different "project names"
        _replace_file('A')
        delA = Packaging('tg_test_alongside_1', '0.0.1', dict(test=_FeaturizerForTestPurposes('argument A')))
        delA.silent = True
        delA.make_package()

        _replace_file('B')
        delB = Packaging('tg_test_alongside_2', '0.0.2', dict(test=_FeaturizerForTestPurposes('argument B')))
        delB.silent = True
        delB.make_package()

        loaderA = install_package_and_get_loader(delA.package_location, silent=False)
        loaderB = install_package_and_get_loader(delB.package_location, silent=False)
        self.loader_expectations(loaderA, 'tg_test_alongside_1__0_0_1')
        self.loader_expectations(loaderB, 'tg_test_alongside_2__0_0_2')

        self.assertEqual('argument A, package A', loaderA.load_resource('test')())
        self.assertEqual('argument B, package B', loaderB.load_resource('test')())

        _replace_file('')
        uninstall_old_version(delA.package_location)
        uninstall_old_version(delB.package_location)
        os.remove(delA.package_location.__str__())
        os.remove(delB.package_location.__str__())





_TEMPLATE = '''

class _FeaturizerForTestPurposes:
    def __init__(self, return_value):
        self.return_value = return_value
        
    def get_suffix(self):
        return ', package {0}'

    def __call__(self, *args, **kwargs):
        return self.return_value+self.get_suffix()
'''
