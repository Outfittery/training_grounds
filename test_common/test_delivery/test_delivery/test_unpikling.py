from tg.common.delivery.delivery.entry_point import HackedUnpicker
from yo_fluq_ds import FileIO
import tg.common.test_common.test_delivery.test_delivery.class_a as class_a
import tg.common.test_common.test_delivery.test_delivery.class_b as class_b
import os

from unittest import TestCase


class UnpicklingTestCase(TestCase):
    def test_ordinary_unpickling(self):
        FileIO.write_pickle(class_a.TestClass(), 'test.pkl')

        ta = FileIO.read_pickle('test.pkl')
        self.assertIsInstance(ta, class_a.TestClass)
        self.assertEqual('A', ta.get_value())

        with open('test.pkl', 'rb') as file_obj:
            tb = HackedUnpicker(file_obj, 'tg.common.test_common.test_delivery.test_delivery.class_a', 'tg.common.test_common.test_delivery.test_delivery.class_b').load()
        self.assertIsInstance(tb, class_b.TestClass)
        self.assertEqual('B', tb.get_value())

        os.remove('test.pkl')
