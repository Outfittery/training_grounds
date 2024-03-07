import os
from unittest import TestCase
import pandas as pd
from tg.common.data import Cache
from tg.common.data.cache import DefaultPickleHandler
from tg.common import Loc
import shutil

def get_df(value):
    return pd.DataFrame(dict(values=[value, value]))

folder = Loc.temp_path/'tests/cache_tests'
Cache.DEFAULT_FOLDER = folder

class CacheTestCase(TestCase):
    def check(self, df, value):
        self.assertListEqual([value, value], list(df.values))

    def test_function(self):
        df = get_df(2)
        self.check(df, 2)

    def test_cache(self):
        shutil.rmtree(folder, ignore_errors=True)
        df = Cache.source(get_df, 1).to('test_cache').get_data()
        self.check(df, 1)
        stored = pd.read_parquet(folder/'test_cache.parquet')
        self.check(stored, 1)

        df = Cache.source(get_df, 2).to('test_cache').get_data()
        self.check(df, 1) #because it will be taken from cache, the name is not autocomposed

        df = Cache.source(get_df, 2).to('test_cache').mode('use').get_data()
        self.check(df, 1)

        df = Cache.source(get_df, 2).to('test_cache').mode('no').get_data()
        self.check(df, 2)
        self.check(pd.read_parquet(folder / 'test_cache.parquet'), 1)

        df = Cache.source(get_df, 2).to('test_cache').mode('remake').get_data()
        self.check(df, 2)
        self.check(pd.read_parquet(folder / 'test_cache.parquet'), 2)

    def test_pickle_cache_and_conflict(self):
        shutil.rmtree(folder, ignore_errors=True)
        df = Cache.source(get_df, 1).to('test_cache').get_data()
        self.check(df, 1)
        self.check(pd.read_parquet(folder / 'test_cache.parquet'), 1)

        self.assertRaises(ValueError,Cache.source(get_df, 2).to('test_cache').via(DefaultPickleHandler()).get_data)

        df = Cache.source(get_df, 2).to('test_cache').via(DefaultPickleHandler()).mode('remake').get_data()
        self.check(df, 2)

        self.assertRaises(ValueError, Cache.source(get_df, 2).to('test_cache').get_data)

    def test_pickle_cache(self):
        shutil.rmtree(folder, ignore_errors=True)
        df = Cache.source(get_df, 1).to('test_cache').via(DefaultPickleHandler()).get_data()
        self.check(df, 1)

        df = Cache.source(get_df, 2).to('test_cache').via(DefaultPickleHandler()).get_data()
        self.check(df, 1)

    def test_autonaming_short(self):
        shutil.rmtree(folder, ignore_errors=True)
        df = Cache.source(get_df, 1).get_data()
        self.assertEqual(['get_df_1.parquet'], os.listdir(folder))

    def test_autonaming_long(self):
        shutil.rmtree(folder, ignore_errors=True)
        df = Cache.source(get_df, '1'*1000).get_data()
        self.assertEqual(['get_df_cac0c3f51680222f14025641074e35fa.parquet'], os.listdir(folder))

