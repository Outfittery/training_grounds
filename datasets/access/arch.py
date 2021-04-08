from typing import *
from yo_fluq_ds import Query, Queryable
import os
import pandas as pd
from ..._common.locations import Loc



# TODO: add short description of module


_CACHE = Loc.data_cache_path.joinpath('downloads')
os.makedirs(_CACHE, exist_ok=True)


class DataSource:
    def get_data(self) -> Queryable:
        """
        Reads data from the source one-by-one and returns as queryable
        """
        raise NotImplementedError()


class MockDfDataSource(DataSource):
    """
    Provides a datasource wrap over pandas dataframe
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_data(self) -> Queryable:
        return Query.df(self.df)

    @staticmethod
    def from_code_list(n_columns, *args):
        column_names = args[:n_columns]
        rows= []
        i=n_columns
        while True:
            row = {}
            broken = False
            for j in range(n_columns):
                if i+j<len(args):
                    row[column_names[j]]=args[i+j]
                else:
                    broken = True
            if broken:
                break
            rows.append(row)
            i+=n_columns
        return MockDfDataSource(pd.DataFrame(rows))



class AbstractCacheDataSource(DataSource):
    """
    Data source that can build a cache from another datasource, store and reproduce it.
    """

    def cache_from(self, src: Queryable, cnt=None):
        """
        Caches data from the datasource
        """
        raise NotImplementedError()

    def is_available(self) -> bool:
        """
        Checks if the cache is available
        """
        raise NotImplementedError()


class CacheableDataSource(DataSource):
    """
    Data source that can cache itself and be later read from this cache
    """

    def __init__(self, inner_data_source: DataSource, file_data_source: AbstractCacheDataSource):
        self._inner_datasource = inner_data_source
        self._file_data_source = file_data_source

    def get_data(self) -> Queryable:
        return self._inner_datasource.get_data()

    def make_cache(self, cnt: Optional[int] = None) -> None:
        self._file_data_source.cache_from(self._inner_datasource.get_data(), cnt)

    def cache(self) -> AbstractCacheDataSource:
        return self._file_data_source

    def safe_cache(self, cache_mode, count: Optional[int] = None):
        if cache_mode == 'no':
            return self
        if cache_mode == 'use' and not self.cache().is_available():
            raise ValueError('Reading from cache is forced, but cache was not available. Try using `default` option')
        if not self.cache().is_available() or cache_mode == 'remake':
            self.make_cache(count)
        return self.cache()
