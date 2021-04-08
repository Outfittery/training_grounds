from unittest import TestCase
from tg.common.ml.miscellaneous import QuantileProportionDifference

class QPDTestCase(TestCase):
    def test_qpd(self):
        predicted = [0.1, 0.9, 0.2, 0.8] + [0.5]*6
        true = [0.1, 0.9, 0.3, 0.7] + [0.5]*6
        qpd = QuantileProportionDifference(0.2)
        self.assertAlmostEqual(0.6,qpd(true,predicted))