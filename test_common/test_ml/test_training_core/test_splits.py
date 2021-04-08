from unittest import TestCase
import pandas as pd
from tg.common.ml.training_core import *
from yo_fluq_ds import *
from datetime import datetime, timedelta


class SplitsTestCase(TestCase):
    def createDfs(self):
        df = pd.Series(list(range(1000))).to_frame('x')
        df['y'] = df.x
        df['z'] = df.x
        df = df.set_index('x')
        dfs = DataFrameSplit(df, ['y'], 'z')
        return dfs

    def test_fold(self):
        dfs = self.createDfs()
        splitter = FoldSplitter(3, 0.3)
        dfs = splitter(dfs)

        self.assertEqual(3, len(dfs))
        set1 = set(dfs[0].train)
        set2 = set(dfs[1].train)
        self.assertLess(400, len(set1.intersection(set2)))
        for c in dfs:
            self.assertEqual(700, len(c.train))
            self.assertEqual(300, len(c.tests['test']))
            self.assertEqual(0, len(set(c.train).intersection(c.tests['test'])))


    def test_custom_fold(self):
        df = pd.Series(list(range(10))).to_frame('x')
        df['y'] = (df.x/2).astype('int')
        dfs = DataFrameSplit(df,['y'],'x')
        splitter = FoldSplitter(10,0.4,custom_split_column='y')
        dfs = splitter(dfs)
        for c in dfs:
            in_train = c.df.loc[c.train]
            in_test = c.df.loc[c.tests['test']]
            self.assertEqual(6, in_train.shape[0])
            self.assertEqual(4, in_test.shape[0])
            intersection = set(in_train.x).intersection(set(in_test.x))
            self.assertEqual(0, len(intersection))


    def test_two_fold(self):
        dfs = self.createDfs()
        spl1 = FoldSplitter(3, 0.2, 'test1')
        spl2 = FoldSplitter(4, 0.3, 'test2')
        spl3 = CompositionSplit(spl1, spl2)
        dfs = spl3(dfs)
        self.assertEqual(12, len(dfs))
        for c in dfs:
            self.assertEqual(560, len(c.train))
            self.assertEqual(200, len(c.tests['test1']))
            self.assertEqual(240, len(c.tests['test2']))
            self.assertEqual(0, len(set(c.train).intersection(c.tests['test1'])))
            self.assertEqual(0, len(set(c.train).intersection(c.tests['test2'])))
            self.assertEqual(0, len(set(c.tests['test1']).intersection(c.tests['test2'])))

    def test_time_split(self):
        df = Query.en(range(140)).select(
            lambda z: dict(ind=z, x=z, y=z, day=datetime(2019, 1, 1) + timedelta(days=z))).to_dataframe()
        df.set_index('ind')
        timeSplit = TimeSplit(
            'day',
            datetime(2019, 2, 20),  # +50 days
            timedelta(days=20),
            timedelta(days=10)
        )
        dfs = DataFrameSplit(df, ['x'], 'y')
        dfs = timeSplit(dfs)

        for i, d in enumerate(dfs):
            self.assertEqual(0, d.train.min())
            self.assertEqual(50 + 20 * i - 1, d.train.max())
            self.assertEqual(50 + 20 * i + 10, d.tests['test'].min())
            self.assertEqual(50 + 20 * i + 10 + 20 - 1, d.tests['test'].max())

    def test_xy(self):
        df = pd.Series(list(range(1000))).to_frame('x')
        for i in range(3):
            df[f'x{i}'] = df.x
        df['y0'] = df.x
        df['y1'] = df.x
        dfs = DataFrameSplit(df, ['x0', 'x1', 'x2'], 'y0')
        x, y = dfs.get_xy(dfs.train)
        self.assertIsInstance(x, pd.DataFrame)
        self.assertListEqual(['x0', 'x1', 'x2'], list(x.columns))
        self.assertIsInstance(y, pd.Series)

    def test_one_time_split(self):
        df = Query.en(range(100)).select(lambda z: dict(tm=datetime(2019,1,1)+timedelta(days=z), x=z, y=z)).to_dataframe()
        dfs = DataFrameSplit(df, ['x'], 'y')

        oneTimeSplit = OneTimeSplit('tm',0.3)
        dfs = oneTimeSplit(dfs) 
        self.assertEqual(1, len(dfs))
        dfs = dfs[0]
        self.assertEqual(70, dfs.train.shape[0])
        self.assertEqual(30, dfs.tests['test'].shape[0])
        self.assertEqual(69, max(dfs.df.loc[dfs.train].x))
        self.assertEqual(70, min(dfs.df.loc[dfs.tests['test']].x))




