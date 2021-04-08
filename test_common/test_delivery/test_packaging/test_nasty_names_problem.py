from unittest import TestCase
from tg.common.test_common.test_delivery.test_packaging.class_a import TestClass
from tg.common.delivery.packaging import PackagingTask, make_package, install_package_and_get_loader
from tg.common import Loc

class NastyNamesTestCase(TestCase):
    def test_nasty_names(self):
        o = TestClass()
        task = PackagingTask('test-package','v23',dict(o=o))
        info = make_package(task)
        entry_point = install_package_and_get_loader(info.path)

