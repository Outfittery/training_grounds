from unittest import TestCase
from tg.common.test_common.test_delivery.test_packaging.class_hmr import _FeaturizerForTestPurposes
from tg.common.delivery.packaging import *
import os
from pathlib import Path
from yo_fluq_ds import FileIO


def _replace_file(suffix):
    path = Path(__file__).parent.joinpath('class_hmr.py')
    FileIO.write_text(_TEMPLATE.format(suffix), path)


class PackagingTestCase(TestCase):
    def test_packaging(self):
        _replace_file('A')

        # this is an "entry point" of the package. It contains, e.g., the object for featurizer
        task = PackagingTask('tg_test', '0.0.0', dict(test=_FeaturizerForTestPurposes('Passed')))

        # Here we make a package from this entry point
        info = make_package(task)

        # Here we load the package, and the entry point from it
        loaded_entry = install_package_and_get_loader(info.path).load_resource('test')

        # Checks that we really managed to read the entry from the package
        self.assertEqual('PassedA', loaded_entry())

        # cleaning-up
        os.remove(info.path.__str__())

    def test_hot_module_replacement(self):
        # We build two packages with different source code
        _replace_file('A')
        infoA = make_package(
            PackagingTask('tg_test', '0.0.1', dict(test=_FeaturizerForTestPurposes('PassedA')))
        )

        _replace_file('B')
        infoB = make_package(
            PackagingTask('tg_test', '0.0.2', dict(test=_FeaturizerForTestPurposes('PassedB')))
        )

        # Loading the first package and checking if it works
        entryA = install_package_and_get_loader(infoA.path)
        self.assertEqual('PassedAA', entryA.load_resource('test')())
        self.assertTrue(os.path.isdir(entryA.resources_location))

        # Hot-Replacing the first package with the second one, and checking if it works
        entryB = install_package_and_get_loader(infoB.path)
        self.assertFalse(os.path.isdir(entryA.resources_location))

        self.assertEqual('PassedBB', entryB.load_resource('test')())
        print('LOCATION ' + entryB.resources_location)

        # Cleaning-up
        _replace_file('')
        os.remove(infoA.path.__str__())
        os.remove(infoB.path.__str__())


_TEMPLATE = '''

class _FeaturizerForTestPurposes:
    def __init__(self, return_value):
        self.return_value = return_value
        
    def get_suffix(self):
        return '{0}'

    def __call__(self, *args, **kwargs):
        return self.return_value+self.get_suffix()
'''
