from tg.common.ml.batched_training.torch import networks as btn
import pandas as pd
from unittest import TestCase

batch = dict()
a = pd.DataFrame(dict(
    a=[0,0],
    b=[0,1],
    c=[1,0],
    d=[1,1]
))
batch['a'] = a

b = pd.DataFrame(dict(
    x=[0,1],
    y=[1,1],
    z=[1,0]
))
batch['b'] = b

class NetworksTestCase(TestCase):
    def check(self, factory, shape):
        network = factory.create_network(None, batch)
        result = network(batch)
        self.assertListEqual(
            list(shape),
            list(result.shape)
        )

    def test_fullyconnected_on_1(self):
        self.check(btn.FullyConnectedNetwork.Factory([10]).prepend_extraction(['a']), [2,10])

    def test_fullyconnected_on_2_with_output(self):
        self.check(btn.FullyConnectedNetwork.Factory([10], output=3).prepend_extraction(['a','b']), [2,3])

    def text_extracting(self):
        self.check(btn.ExtractingNetwork.Factory(['a']), [2,4])

    def test_parallel(self):
        factory = (
            btn.ParallelNetwork.Factory(
                a=btn.FullyConnectedNetwork.Factory([5]).prepend_extraction('a'),
                b=btn.FullyConnectedNetwork.Factory([7]).prepend_extraction('b')
            ))
        network = factory.create_network(None, batch)
        result = network(batch)
        self.assertEqual((2,5), result['a'].shape)
        self.assertEqual((2,7), result['b'].shape)


    def test_missing_in_forgiving_extractor(self):
        self.check(btn.ExtractingNetwork.Factory(['a','x']), [2, 4])

