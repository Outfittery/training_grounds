from unittest import TestCase
from tg.common.ml.batched_training.context import lstm_data_transformation
from tg.common.ml.batched_training.torch import AnnotatedTensor, DfConversion
import pandas as pd
import torch


def cv(t):
    return [round(c, 2) for c in t.tolist()]


class LSTMContextTestCase(TestCase):
    def check(self, q: AnnotatedTensor):
        self.assertEqual(('context', 'sample_id', 'features'), q.dim_names)
        self.assertEqual((1, 2, 3), q.dim_indices[0])
        self.assertEqual((10, 20), q.dim_indices[1])
        self.assertEqual(('f1', 'f2', 'f3', 'f4'), q.dim_indices[2])
        for i in range(3):
            self.assertEqual(len(q.dim_indices[i]), q.tensor.shape[i])
        self.assertListEqual([0.2, 0.1, 0.1, 0.1], cv(q.tensor[0, 0, :]))
        self.assertListEqual([0.1, 0.2, 0.1, 0.1], cv(q.tensor[1, 0, :]))
        self.assertListEqual([0.4, 0.4, 0.4, 0.5], cv(q.tensor[0, 1, :]))

    def test_simple(self):
        tdf = pd.DataFrame([
            (10, 1, 0.2, 0.1, 0.1, 0.1),
            (10, 2, 0.1, 0.2, 0.1, 0.1),
            (10, 3, 0.1, 0.1, 0.2, 0.1),
            (20, 1, 0.4, 0.4, 0.4, 0.5),
            (20, 2, 0.4, 0.4, 0.5, 0.4),
            (20, 3, 0.4, 0.5, 0.4, 0.4),
        ], columns=['sample_id', 'context', 'f1', 'f2', 'f3', 'f4']).set_index(['sample_id', 'context'])
        index = pd.Index([10, 20])
        q = lstm_data_transformation(index, [1, 2, 3], tdf, DfConversion.float)
        self.check(q)

    def test_reordering(self):
        tdf = pd.DataFrame([
            (20, 3, 0.4, 0.5, 0.4, 0.4),
            (20, 2, 0.4, 0.4, 0.5, 0.4),
            (20, 1, 0.4, 0.4, 0.4, 0.5),
            (10, 3, 0.1, 0.1, 0.2, 0.1),
            (10, 2, 0.1, 0.2, 0.1, 0.1),
            (10, 1, 0.2, 0.1, 0.1, 0.1),
        ], columns=['sample_id', 'context', 'f1', 'f2', 'f3', 'f4']).set_index(['sample_id', 'context'])
        index = pd.Index([10, 20])
        q = lstm_data_transformation(index, [1, 2, 3], tdf, DfConversion.float)
        self.check(q)

    def test_missing(self):
        tdf = pd.DataFrame([
            (10, 1, 0.2, 0.1, 0.1, 0.1),
            (10, 2, 0.1, 0.2, 0.1, 0.1),
            (20, 1, 0.4, 0.4, 0.4, 0.5),
            (20, 3, 0.4, 0.5, 0.4, 0.4),
        ], columns=['sample_id', 'context', 'f1', 'f2', 'f3', 'f4']).set_index(['sample_id', 'context'])
        index = pd.Index([10, 20])
        q = lstm_data_transformation(index, [1, 2, 3], tdf, DfConversion.float)
        self.check(q)
        self.assertListEqual([0, 0, 0, 0], cv(q.tensor[2, 0, :]))
        self.assertListEqual([0, 0, 0, 0], cv(q.tensor[1, 1, :]))

    def test_int(self):
        tdf = pd.DataFrame([
            (10, 1, 101, 201),
            (10, 2, 102, 202)
        ], columns=['sample_id','context','f1','f2']).set_index(['sample_id','context'])
        index = pd.Index([10])
        q = lstm_data_transformation(index, [1,2,3], tdf, DfConversion.int)
        self.assertListEqual([101,201], list(q.tensor[0,0,:]))
        self.assertListEqual([0, 0], list(q.tensor[2, 0, :]))



    def test_torch_sampling(self):
        t = torch.Tensor([
            [
                [111, 112, 113, 114, ],
                [121, 122, 123, 124],
                [131, 132, 133, 134]
            ],
            [
                [211, 212, 213, 214, ],
                [221, 222, 223, 224],
                [231, 232, 233, 234]
            ]
        ])
        at = AnnotatedTensor(
            t,
            ['a', 'b', 'c'],
            [
                ['a1', 'a2'],
                ['b1', 'b2', 'b3'],
                ['c1', 'c2', 'c3', 'c4']
            ]
        )
        s = at.sample_index(pd.Index(['b2', 'b3'], name='b'))
        self.assertListEqual([2, 2, 4], list(s.tensor.shape))
