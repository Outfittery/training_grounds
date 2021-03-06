from tg.common.ml.batched_training import *
from tg.common import Loc
from unittest import TestCase
import os
import shutil

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
            df1 = [str(a) for a in range(100, 110)],
            df2 = [a for a in range(10, 20)]
        )
    )
    return DataBundle(index,dict(df1=df1,df2=df2)).as_indexed(index.index[[1,3,5,7]])


class BatcherTestCase(TestCase):
    def test_batching_strategy(self):
        db = get_bundle()
        strategy = SimpleBatcherStrategy()
        idx = strategy.get_batch(2, db.bundle.index_frame.loc[db.index], 0)
        self.assertIsInstance(idx, pd.Int64Index)
        self.assertListEqual([1,3], list(idx))


    def test_batching(self):
        db = get_bundle()
        strategy = SimpleBatcherStrategy()
        batcher = Batcher(2,[
            DirectExtractor('df1'),
            DirectExtractor('df2')
            ])
        self.assertEqual(2, batcher.get_batch_count(db, strategy))
        batch = batcher.get_batch(db,1, strategy)
        self.assertListEqual([5,7],list(batch['df1'].a))
        self.assertListEqual([5, 7], list(batch['df1'].index))
        self.assertListEqual(['-5', '-3'], list(batch['df2'].b))
        self.assertListEqual([5, 7], list(batch['df2'].index))

        self.assertListEqual([5,7], list(batch['index'].index))
        self.assertListEqual(['105', '107'], list(batch['index'].df1))
        self.assertListEqual([15,17], list(batch['index'].df2))


    def test_loading(self):
        bundle = get_bundle().bundle
        folder = Loc.temp_path.joinpath('tests/dbbundle')
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        bundle.save(folder)
        bundle1 = DataBundle.ensure(folder)
        self.assertSetEqual({'df1','df2'}, set(bundle1.data_frames.keys()))
        self.assertListEqual(['a'],list(bundle1.data_frames['df1'].columns))
        self.assertListEqual(['b'], list(bundle1.data_frames['df2'].columns))


    def _unwrap(self, batcher, db, strategy):
        result = []
        for i in range(batcher.get_batch_count(db, strategy)):
            batch = batcher.get_batch(db,i, strategy)
            result.extend(list(batch['df1']['a']))
        return result


    def test_uneven_batcch(self):
        strategy = SimpleBatcherStrategy()
        batcher = Batcher(2, [
            DirectExtractor('df1'),
            DirectExtractor('df2')
            ])

        db = get_bundle()
        db.index = db.bundle.index_frame.index[[0,1,2,3]]
        self.assertEqual(2,batcher.get_batch_count(db, strategy))
        self.assertListEqual([0,1,2,3],self._unwrap(batcher,db, strategy))

        db.index = db.bundle.index_frame.index[[0,1,2,3,4]]
        self.assertEqual(3,batcher.get_batch_count(db, strategy))
        self.assertListEqual([0, 1, 2, 3, 4], self._unwrap(batcher, db, strategy))




