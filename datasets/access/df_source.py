from typing import *
import pandas as pd
from .arch import CacheMode, DataSource
from pathlib import Path
import os
from ..._common import DataBundle

class DataFrameSource:
    def get_df(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_default_filename(self):
        return None

    def get_cached_df(self, filename: Optional[Union[Path,str]] = None, cache_mode: Union[str,CacheMode] = 'default'):
        if filename is None:
            filename = self.get_default_filename()
            if filename is None:
                raise NotImplementedError('The filename was not provided and also cannot be generated automatically')
        return CacheMode.apply_to_file(
            cache_mode,
            filename,
            self.get_df
        )

class DataFrameSourceOverDataSource(DataFrameSource):
    def __init__(self, src: DataSource):
        self.src = src

    def get_df(self) -> pd.DataFrame:
        return self.src.get_data().to_dataframe()


class DataBundleSourceLoader:
    def __init__(self, location: Path, **sources: DataFrameSource):
        self.location = location
        self.sources = sources

    def download(self, cache_mode='default'):
        os.makedirs(str(self.location), exist_ok=True)
        dfs = {}
        for key, source in self.sources.items():
            dfs[key] = source.get_cached_df(
                self.location / key,
                cache_mode
            )
        return DataBundle(**dfs)


class InMemoryDataFrameSource(DataFrameSource):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_df(self) -> pd.DataFrame:
        return self.df


class LambdaDataFrameSource(DataFrameSource):
    def __init__(self, method, *args, **kwargs):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def get_df(self) -> pd.DataFrame:
        return self.method(*self.args, **self.kwargs)