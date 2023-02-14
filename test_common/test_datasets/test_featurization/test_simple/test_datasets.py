from unittest import TestCase
import os
import shutil
from tg.common import Loc
import pandas as pd
import inspect
from tg.common.datasets.featurization import *
from datetime import datetime, timedelta

TIME = datetime.now() - timedelta(days=30)


def DT(days):
    return TIME + timedelta(days=days)


class Preparer:
    def __init__(self):
        self.testname = inspect.stack()[1][3]
        self.folder = Loc.temp_path / 'tests' / self.testname
        shutil.rmtree(self.folder, True)
        os.makedirs(self.folder)
        self.counter = 0
        self.time_table = []

    def add(self, path, *index) -> 'Preparer':
        rows = []
        for i in index:
            rows.append(dict(ind=i, a=i, b=i * 10, cnt=self.counter))
            self.counter += 1
        df = pd.DataFrame(rows)
        df = df.set_index('ind')
        fname = self.folder / (path + ".parquet")
        os.makedirs(fname.parent, exist_ok=True)
        df.to_parquet(fname)
        return self

    def at(self, name, delta, is_major):
        self.time_table.append(UpdatableDataset.DescriptionItem(
            name,
            TIME + timedelta(days=delta),
            is_major,
            '1'
        ))
        return self

    def ready(self):
        if len(self.time_table) > 0:
            UpdatableDataset.DescriptionHandler.write_parquet(self.time_table, self.folder / 'description.parquet')
        return self.folder


class LocalTestCase(TestCase):
    def test_dataset_reading(self):
        path = Preparer().add('1', 1, 2, 3).add('2', 5, 6).ready()
        ds = Dataset(path, None)
        self.assertListEqual([1, 2, 3, 5, 6], list(ds.read().index))
        self.assertListEqual([1, 2], list(ds.read(count=2).index))
        self.assertListEqual([2, 6], list(ds.read(selector=lambda z: z.loc[z.index % 2 == 0]).index))
        self.assertListEqual(['a', 'cnt'], list(ds.read(columns=['a', 'cnt']).columns))

    def test_partioned_dataset(self):
        path = (
            Preparer()
                .add('a/x/1', 1, 2, 3)
                .add('a/x/2', 7, 8)
                .add('b/x/1', 2, 7, 9)
                .add('c/x/1', 2, 10)
                .at('a', 0, True)
                .at('b', 2, False)
                .at('c', 4, False)
                .ready()
        )
        ds = UpdatableDataset(path, 'x', None)
        self.assertListEqual([8, 9, 6, 7, 0, 2, 4], list(ds.read(cache_mode='use').cnt))
        self.assertListEqual([2, 10, 7, 9, 1, 3, 8], list(ds.read(cache_mode='use').index))
        self.assertListEqual([2, 10, 7, 9], list(ds.read(count=4, cache_mode='use').index))

        df = ds.read(from_timestamp=DT(1), cache_mode='use')
        self.assertListEqual([8, 9, 6, 7], list(df.cnt))

        df = ds.read(from_timestamp=DT(1), to_timestamp=DT(3), cache_mode='use')
        self.assertListEqual([5, 6, 7], list(df.cnt))

    def test_double_major(self):
        path = (
            Preparer()
                .add('a/x/1', 1, 2, 3)
                .add('b/x/1', 2, 3)
                .add('c/x/1', 2, 4, 5)
                .add('d/x/1', 3, 5)
                .at('a', 0, True)
                .at('b', 2, False)
                .at('c', 4, True)
                .at('d', 6, False)
                .ready()
        )

        ds = UpdatableDataset(path, 'x', None)
        self.assertListEqual([8, 9, 5, 6], list(ds.read(cache_mode='use').cnt))

        df = ds.read(from_timestamp=DT(2), cache_mode='use')
        self.assertListEqual([8, 9, 5, 6], list(df.cnt))

        df = ds.read(to_timestamp=DT(3), cache_mode='use')
        self.assertListEqual([3, 4, 0], list(df.cnt))
