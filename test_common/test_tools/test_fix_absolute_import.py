from unittest import TestCase
from tg.common.tools_bak.fix_absolute_import import *

class FixAbsolutePathTestCase(TestCase):
    def test_relative_module_name(self):
        t = build_replacement(
            '/home/repo/tg',
            '/home/repo/tg/common/datasets/selectors/combinators',
            'tg.common.datasets.access'
        )
        self.assertEqual(
            '..access',
            t
        )

    def test_module_pattern(self):
        self.assertEqual(
            'tg.common.datasets.something',
            find_module_name('from  tg.common.datasets.something import *')
        )
        self.assertEqual(
            None,
            find_module_name('from  sklearn import')
        )

