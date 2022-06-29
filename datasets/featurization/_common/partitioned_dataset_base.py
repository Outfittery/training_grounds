from typing import *

import os
import shutil
import pandas as pd

from datetime import datetime
from pathlib import Path
from yo_fluq_ds import Query, Queryable, fluq
from uuid import uuid4

from ...._common import FileSyncer, Loc
from ...access import CacheMode
from ..simple.dataset import Dataset, LambdaDataFrameSource
from .index_filters import SimpleIndexFilter


T = TypeVar('T')


class PartitionedDatasetRecordHandler(Generic[T]):
    def __init__(self, description_filename, type, name_field):
        self.description_filename = description_filename
        self.type = type
        self.name_field = name_field

    def get_description_filename(self) -> str:
        return self.description_filename

    def read_parquet(self, filename) -> List[T]:
        return Query.df(pd.read_parquet(filename)).select(lambda z: self.type(**z)).to_list()

    def write_parquet(self, records: List[T], filename):
        Query.en(records).select(lambda z: z.__dict__).to_dataframe().to_parquet(filename)

    def write_parquet_to_folder(self, records: List[T], folder):
        self.write_parquet(records, folder / self.get_description_filename())

    def get_name_from_record(self, record: T):
        return getattr(record, self.name_field)


class TimePartitionedDatasetBase(Generic[T]):
    def __init__(self,
                 location: Union[Path, str],
                 featurizer_name: str,
                 syncer: FileSyncer,
                 record_handler: PartitionedDatasetRecordHandler[T]
                 ):
        self.location = Path(location)
        self.featurizer_name = featurizer_name
        self.record_handler = record_handler
        if syncer is not None:
            self.syncer = syncer.change_local_folder(self.location)
        else:
            self.syncer = None

    def filter_relevant_records(self, records: List[T], from_time: datetime, to_time: datetime) -> List[T]:
        raise NotImplementedError()

    def get_description(self) -> List[T]:
        return self.record_handler.read_parquet(self.location / self.record_handler.get_description_filename())

    def get_desription_as_df(self) -> pd.DataFrame:
        return pd.read_parquet(self.location / self.record_handler.get_description_filename())

    def spawn_child_dataset(self, partition_name):
        return Dataset(
            self.location / partition_name / self.featurizer_name,
            None if self.syncer is None else self.syncer.cd(os.path.join(partition_name, self.featurizer_name))
        )

    def _update_index(self):
        self.syncer.download_file(self.record_handler.get_description_filename())

    def download(self,
                 from_timestamp: Optional[datetime] = None,
                 to_timestamp: Optional[datetime] = None,
                 cache_mode: Union[str, CacheMode, None] = CacheMode.Default,
                 with_progress_bar: bool = False,
                 ):
        cache_mode = CacheMode.parse(cache_mode)
        if cache_mode == CacheMode.No:
            cache_mode = CacheMode.Remake

        if cache_mode == CacheMode.Remake:
            shutil.rmtree(self.location, True)

        os.makedirs(self.location, exist_ok=True)
        if cache_mode != CacheMode.Use:
            self._update_index()

        records = self.get_description()
        records = self.filter_relevant_records(records, from_timestamp, to_timestamp)

        if cache_mode == CacheMode.Use:
            return records

        datasets_to_download = []
        for rec in records:
            child = self.spawn_child_dataset(self.record_handler.get_name_from_record(rec))
            if not child.is_available() or cache_mode == CacheMode.Remake:
                datasets_to_download.append(child)

        download_query = Query.en(datasets_to_download)
        if with_progress_bar:
            download_query = download_query.feed(fluq.with_progress_bar())

        for child in download_query:
            child.download()

        return records

    def _read_iter(self,
                   from_timestamp: Optional[datetime] = None,
                   to_timestamp: Optional[datetime] = None,
                   columns: Optional[List] = None,
                   selector: Optional[Callable] = None,
                   count: Optional[int] = None,
                   partition_name_column: Optional[str] = None,
                   with_progress_bar: bool = False
                   ):
        seen_index = None
        collected = 0
        records = self.filter_relevant_records(self.get_description(), from_timestamp, to_timestamp)
        records_query = Query.en(records)
        if with_progress_bar:
            records_query = records_query.feed(fluq.with_progress_bar())
        filter = SimpleIndexFilter()
        for dset in records_query:
            for df in self.spawn_child_dataset(dset.name).read_iter(columns, selector, None):
                df, seen_index = filter.filter(df, seen_index)
                df = df.copy()
                if partition_name_column is not None:
                    df[partition_name_column] = self.record_handler.get_name_from_record(dset)
                if count is None:
                    yield df
                else:
                    remaining = count - collected
                    if remaining <= 0:
                        break
                    if remaining < df.shape[0]:
                        df = df.iloc[:remaining].copy()
                    collected += df.shape[0]
                    yield df

    def read_iter(self,
                  from_timestamp: Optional[datetime] = None,
                  to_timestamp: Optional[datetime] = None,
                  columns: Optional[List] = None,
                  selector: Optional[Callable] = None,
                  count: Optional[int] = None,
                  partition_name_column: Optional[str] = None,
                  with_progress_bar: bool = False
                  ) -> Queryable:
        return Queryable(
            self._read_iter(from_timestamp, to_timestamp, columns, selector, count, partition_name_column, with_progress_bar)
        )

    def read(self,
             from_timestamp: Optional[datetime] = None,
             to_timestamp: Optional[datetime] = None,
             columns: Optional[List] = None,
             selector: Optional[Callable] = None,
             count: Optional[int] = None,
             partition_name_column: Optional[str] = None,
             cache_mode: Optional[Union[str, CacheMode]] = CacheMode.Default,
             with_progress_bar: bool = False
             ):
        self.download(from_timestamp, to_timestamp, cache_mode, with_progress_bar)
        dfs = self.read_iter(from_timestamp, to_timestamp, columns, selector, count, partition_name_column, with_progress_bar).to_list()
        df = pd.concat(dfs, sort=False)
        return df

    def as_data_frame_source(self, **kwargs):
        return LambdaDataFrameSource(self.read, **kwargs)
