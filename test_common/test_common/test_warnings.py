from unittest import TestCase
from tg.common._common.warnings import *
import pandas as pd
import numpy as np


class TGWarningTestCase(TestCase):
    def test_grouping(self):
        center = TGWarningStorageClass()
        center.add_warning('1', dict(a=1, b=2))
        center.add_warning('1', dict(a=1, b=3))
        center.add_warning('1', dict(a=1, b=3))
        center.add_warning('2', dict(a=2))
        df = pd.DataFrame(center.get_report(1))
        self.assertListEqual(['1', '1', '2'], list(df._message))
        self.assertListEqual([1, 1, 2], list(df.a))
        self.assertListEqual([2.0, 3.0, -1.0], list(df.b.fillna(-1)))
        self.assertListEqual([1, 2, 1], list(df._count))

    def test_levels(self):
        center = TGWarningStorageClass()
        center.add_warning('1', dict(a=1), dict(b=10))
        center.add_warning('1', dict(a=1), dict(b=20))
        center.add_warning('1', dict(a=2), dict(b=20))

        df = pd.DataFrame(center.get_report(0))
        self.assertEqual(1, df.shape[0])
        self.assertListEqual(['1'], list(df._message))
        self.assertListEqual([3], list(df._count))

        df = pd.DataFrame(center.get_report(1))
        self.assertEqual(2, df.shape[0])
        self.assertListEqual(['1', '1'], list(df._message))
        self.assertListEqual([2, 1], list(df._count))
        self.assertListEqual([1, 2], list(df.a))

        df = pd.DataFrame(center.get_report(2))
        self.assertEqual(3, df.shape[0])
        self.assertListEqual(['1', '1', '1'], list(df._message))
        self.assertListEqual([1, 1, 1], list(df._count))
        self.assertListEqual([1, 1, 2], list(df.a))
        self.assertListEqual([10, 20, 20], list(df.b))

        self.assertEqual(3, len(center.get_report()))

    def test_none(self):
        center = TGWarningStorageClass()
        center.add_warning('1', dict(a=None))
        r = center.get_report()
        self.assertEqual(None, r[0]['a'])
        self.assertEqual('1', r[0]['_message'])
        self.assertEqual(1, r[0]['_count'])
