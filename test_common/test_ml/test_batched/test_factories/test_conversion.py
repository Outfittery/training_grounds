import pandas as pd
import torch

from tg.common.ml.batched_training.factories import DfConversion
from unittest import TestCase

df = pd.DataFrame(dict(
    float_1 = [1.1, 2.2],
    float_2 = [0.1, 0.2],
    int_1 = [1, 2],
    int_2 = [10, 20]
))

class DfConversionTestCase(TestCase):
    def test_float(self):
        t = DfConversion.float(df)
        self.assertEqual(torch.float, t.dtype)

    def test_int(self):
        t = DfConversion.int(df)
        self.assertEqual(torch.int32, t.dtype)

    def test_auto(self):
        t = DfConversion.auto(df)
        self.assertEqual(torch.float, t.dtype)

    def test_auto_on_int_only(self):
        t = DfConversion.auto(df[['int_1', 'int_2']])
        self.assertEqual(torch.int32, t.dtype)