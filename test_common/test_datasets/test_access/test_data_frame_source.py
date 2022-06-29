from tg.common.datasets.access import InMemoryDataFrameSource, CacheMode, Loc
from unittest import TestCase
import pandas as pd
import shutil
import os


class DataFrameSourceTestCase(TestCase):
    def test_dataframesource_cache(self):
        df = pd.DataFrame(dict(a=[1, 2, 3], b=[4, 5, 6]))
        fname = Loc.temp_path / 'tests/data_frame_source/data.parquet'
        shutil.rmtree(fname.parent, ignore_errors=True)
        os.makedirs(fname.parent)

        src = InMemoryDataFrameSource(df)
        src.get_cached_df(fname, CacheMode.No)
        self.assertFalse(fname.is_file())
        src.get_cached_df(fname, CacheMode.Default)
        self.assertTrue(fname.is_file())
        src.df = None
        rdf = src.get_cached_df(fname, CacheMode.Default)
