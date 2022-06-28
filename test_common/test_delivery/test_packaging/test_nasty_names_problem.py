from unittest import TestCase
from tg.common.test_common.test_delivery.test_packaging.class_a import TestClass
from tg.common.delivery.packaging import PackagingTask, make_package, install_package_and_get_loader
from tg.common.delivery.packaging.package import _get_old_module_to_remove, uninstall_old_version
from tg.common import Loc
import subprocess
import sys

class NastyNamesTestCase(TestCase):
    def test_nasty_names(self):
        o = TestClass()
        task = PackagingTask('test-package','v23',dict(o=o))
        info = make_package(task)
        entry_point = install_package_and_get_loader(info.path)

        deps = subprocess.check_output([sys.executable,'-m','pip','freeze']).decode('utf-8')
        module_name = _get_old_module_to_remove(info.path)+' @'
        self.assertIn(module_name, deps)

        uninstall_old_version(info.path)
        deps = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')
        self.assertNotIn(module_name, deps)


