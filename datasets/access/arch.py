from typing import *

import os
import pandas as pd

from yo_fluq_ds import Query, Queryable, FileIO
from enum import Enum
from pathlib import Path


from ..._common.locations import Loc
from ..._common import DataBundle


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
        rows = []
        i = n_columns
        while True:
            row = {}
            broken = False
            for j in range(n_columns):
                if i + j < len(args):
                    row[column_names[j]] = args[i + j]
                else:
                    broken = True
            if broken:
                break
            rows.append(row)
            i += n_columns
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

class _FileApplicator:
    def __init__(self, read_lambda, write_lambda):
        self.read_lambda = read_lambda
        self.write_lambda = write_lambda

    def get(self, file_path: Union[str, Path], factory: Callable, cache_mode: Union['CacheMode',str] = 'default'):
        val = CacheMode.parse(cache_mode)
        if val == CacheMode.Use:
            return self.read_lambda(file_path)
        if val == CacheMode.No:
            return factory()
        if val == CacheMode.Remake or not os.path.isfile(file_path):
            obj = factory()
            self.write_lambda(obj, file_path)
            return obj
        return self.read_lambda(file_path)


class CacheMode(Enum):
    """
    Contains the modes for the cacheable data source.
    `No` means the cache will not be used at all.
    `Default` means the cache will be read if available, and created otherwise
    `Use` means the cache will be read if available, but if not, the exception is going to be raised
    `Remake` means the cache will be recreated regardless of its existence
    """
    No = 0
    Default = 1
    Use = 2
    Remake = 3

    @staticmethod
    def apply_to_file(cache_mode: 'CacheMode', file_path: Union[str, Path], factory: Callable):
        if cache_mode == CacheMode.No or cache_mode == 'no':
            return factory()
        if cache_mode == CacheMode.Use or cache_mode == 'use':
            return FileIO.read_pickle(file_path)
        if cache_mode == CacheMode.Remake or not Path(file_path).is_file() or cache_mode == 'remake':
            val = factory()
            FileIO.write_pickle(val, file_path)
            return val
        return FileIO.read_pickle(file_path)

    @staticmethod
    def from_pickle_file():
        return _FileApplicator(FileIO.read_pickle, FileIO.write_pickle)

    @staticmethod
    def from_parquet_file():
        return _FileApplicator(pd.read_parquet, lambda df, fname: df.to_parquet(fname))

    @staticmethod
    def from_bundle_folder():
        return _FileApplicator(DataBundle.load, lambda db, fname: db.save(fname))

    @staticmethod
    def parse(value: Union[None, str, 'CacheMode']) -> 'CacheMode':
        if value is None:
            return CacheMode.Default
        elif isinstance(value, CacheMode):
            return value
        elif value == 'default':
            return CacheMode.Default
        elif value == 'no':
            return CacheMode.No
        elif value == 'use':
            return CacheMode.Use
        elif value == 'remake':
            return CacheMode.Remake
        raise ValueError(f'Cannot recognize value {value}')


class CacheableDataSource(DataSource):
    """
    Data source that can cache itself and be later read from this cache
    """

    def __init__(self, inner_data_source: DataSource, file_data_source: AbstractCacheDataSource, default_mode: Optional[CacheMode] = None):
        self._inner_datasource = inner_data_source
        self._file_data_source = file_data_source
        self._default_mode = default_mode if default_mode is not None else CacheMode.Default

    def get_data(self) -> Queryable:
        return self.safe_cache(self._default_mode).get_data()

    def make_cache(self, cnt: Optional[int] = None) -> None:
        self._file_data_source.cache_from(self._inner_datasource.get_data(), cnt)

    def cache(self) -> AbstractCacheDataSource:
        return self._file_data_source

    def safe_cache(self, cache_mode: Union[str, CacheMode], count: Optional[int] = None) -> DataSource:
        if cache_mode == 'no' or cache_mode == CacheMode.No:
            return self._inner_datasource
        if (cache_mode == 'use' or cache_mode == CacheMode.Use) and not self.cache().is_available():
            raise ValueError('Reading from cache is forced, but cache was not available. Try using `default` option')
        if not self.cache().is_available() or cache_mode == 'remake' or cache_mode == CacheMode.Remake:
            self.make_cache(count)
        return self._file_data_source
