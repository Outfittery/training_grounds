from tg.common.ml.batched_training import *
from tg.common import Loc
from unittest import TestCase
import os
import shutil
import pandas as pd


def get_bundle() -> IndexedDataBundle:
    df1 = pd.DataFrame(
        dict(a=list(range(10))),
        index=[str(a) for a in range(100, 110)]
    )
    df2 = pd.DataFrame(
        dict(b=[str(c) for c in range(-10, 0)]),
        index=[a for a in range(10, 20)]
    )
    index = pd.DataFrame(
        dict(
            df1=[str(a) for a in range(100, 110)],
            df2=[a for a in range(10, 20)]
        )
    )
    return IndexedDataBundle(index.iloc[[1, 3, 5, 7]], DataBundle(index=index, df1=df1, df2=df2))


class BatcherTestCase(TestCase):
    def test_batching_strategy(self):
        db = get_bundle()
        strategy = SequencialSampler()
        idx = strategy.get_batch_index_frame(2, db, 0)
        self.assertIsInstance(idx, pd.DataFrame)
        self.assertListEqual([1, 3], list(idx.index))

    def test_batching(self):
        db = get_bundle()
        strategy = SequencialSampler()
        batcher = Batcher([
            PlainExtractor.build('df1').join('df1', 'df1').apply(),
            PlainExtractor.build('df2').join('df2', 'df2').apply(),
            ])
        self.assertEqual(2, batcher.get_batch_count(2, db, strategy))
        batch = batcher.get_batch(2, db, 1, strategy)
        self.assertListEqual([5, 7], list(batch['df1'].a))
        self.assertListEqual([5, 7], list(batch['df1'].index))
        self.assertListEqual(['-5', '-3'], list(batch['df2'].b))
        self.assertListEqual([5, 7], list(batch['df2'].index))

        self.assertListEqual([5, 7], list(batch['index'].index))
        self.assertListEqual(['105', '107'], list(batch['index'].df1))
        self.assertListEqual([15, 17], list(batch['index'].df2))

    def test_loading(self):
        bundle = get_bundle().bundle
        folder = Loc.temp_path.joinpath('tests/dbbundle')
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        bundle.save(folder)
        bundle1 = DataBundle.load(folder)
        self.assertSetEqual({'index', 'df1', 'df2'}, set(bundle1.data_frames.keys()))
        self.assertListEqual(['a'], list(bundle1.data_frames['df1'].columns))
        self.assertListEqual(['b'], list(bundle1.data_frames['df2'].columns))

    def _unwrap(self, batch_size, batcher: Batcher, db, strategy):
        result = []
        for i in range(batcher.get_batch_count(batch_size, db, strategy)):
            batch = batcher.get_batch(batch_size, db, i, strategy)
            result.extend(list(batch['df1']['a']))
        return result

    def test_uneven_batcch(self):
        strategy = SequencialSampler()
        batcher = Batcher([
            PlainExtractor.build('df1').join('df1', 'df1').apply(),
            PlainExtractor.build('df2').join('df2', 'df2').apply(),
            ])

        db = get_bundle()
        db.index_frame = db.bundle.index.iloc[[0, 1, 2, 3]]
        self.assertEqual(2, batcher.get_batch_count(2, db, strategy))
        self.assertListEqual([0, 1, 2, 3], self._unwrap(2, batcher, db, strategy))

        db.index_frame = db.bundle.index.iloc[[0, 1, 2, 3, 4]]
        self.assertEqual(3, batcher.get_batch_count(2, db, strategy))
        self.assertListEqual([0, 1, 2, 3, 4], self._unwrap(2, batcher, db, strategy))
