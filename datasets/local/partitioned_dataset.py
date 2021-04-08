from typing import *

import os
import shutil
import pandas as pd

from datetime import datetime
from pathlib import Path
from yo_fluq_ds import Query, Queryable

from .dataset import Dataset
from .downloader import AbstractDatasetDownloader



class PartitionedDatasetRecord:
    def __init__(self, name: str, timestamp: datetime, is_major: bool, version: str):
        self.name = name
        self.timestamp = timestamp
        self.is_major = is_major
        self.version = version

    @staticmethod
    def write_as_parquet(records: List['PartitionedDatasetRecord'], filename: Path):
        df = pd.DataFrame([c.__dict__ for c in records])
        filename = Path(filename)
        os.makedirs(filename.parent, exist_ok=True)
        df.to_parquet(filename, allow_truncated_timestamps=True)

    @staticmethod
    def read_parquet(filename: Path) -> List['PartitionedDatasetRecord']:
        df = pd.read_parquet(filename)
        return Query.df(df).select(lambda z: PartitionedDatasetRecord(**z)).to_list()


class PartitionedDataset:
    def __init__(self, datasets_location: Union[Path, str], subpath: str, featurizer_name: str,
                 downloader: Optional[AbstractDatasetDownloader] = None):
        self.common_location = Path(datasets_location)
        self.subpath = subpath
        self.downloader = downloader
        self.featurizer_name = featurizer_name
        self._folder_location = self.common_location / self.subpath

    DESCRIPTION_FILENAME = 'description.parquet'

    def get_description(self) -> List[PartitionedDatasetRecord]:
        return PartitionedDatasetRecord.read_parquet(self._folder_location / PartitionedDataset.DESCRIPTION_FILENAME)

    @staticmethod
    def _get_current_records(records: List[PartitionedDatasetRecord], from_timestamp: Optional[datetime],
                             to_timestamp: datetime):
        records = Query.en(records).where(lambda z: z.timestamp <= to_timestamp).order_by_descending(
            lambda z: z.timestamp).to_list()
        first_major = Query.en(records).with_indices().where(lambda z: z.value.is_major).select(
            lambda z: z.key).first_or_default()
        if first_major is None:
            raise ValueError(f"There are no major revisions before {to_timestamp}")
        records = records[:first_major + 1]
        if from_timestamp is not None:
            records = [r for r in records if r.timestamp >= from_timestamp]
        return records

    def get_current_records(self, from_timestamp: Optional[datetime], to_timestamp: Optional[datetime]) -> List[
        PartitionedDatasetRecord]:
        records = self.get_description()
        if to_timestamp is None:
            to_timestamp = datetime.now()
        return PartitionedDataset._get_current_records(records, from_timestamp, to_timestamp)

    def spawn_child_dataset(self, partition_name):
        print(self.subpath, partition_name, self.featurizer_name)
        return Dataset(self.common_location, os.path.join(self.subpath, partition_name, self.featurizer_name),
                       self.downloader)

    def update_index(self):
        if self.downloader is not None:
            self.downloader.download_file(self.common_location, os.path.join(self.subpath, 'description.parquet'))

    def download(self, from_timestamp: Optional[datetime] = None, to_timestamp: Optional[datetime] = None,
                 reporting=None):
        shutil.rmtree(self._folder_location, True)
        os.makedirs(self._folder_location)
        self.update_index()
        if self.downloader is not None:
            records = self.get_current_records(from_timestamp, to_timestamp)
            for rec in records:
                self.spawn_child_dataset(rec.name).download(reporting)

    def _read_iter(self,
                   from_timestamp: Optional[datetime] = None,
                   to_timestamp: Optional[datetime] = None,
                   columns: Optional[List] = None,
                   selector: Optional[Callable] = None,
                   count: Optional[int] = None,
                   update_timestamp_column: Optional[str] = None):
        seen_index = set()
        collected = 0
        records = self.get_current_records(from_timestamp, to_timestamp)
        for dset in records:
            for df in self.spawn_child_dataset(dset.name).read_iter(columns, selector, None):
                df = df.loc[~df.index.isin(seen_index)]
                if update_timestamp_column is not None:
                    df[update_timestamp_column] = dset.timestamp
                seen_index = seen_index.union(df.index)

                if count is None:
                    yield df.copy()
                else:
                    remaining = count - collected
                    if remaining <= 0:
                        break
                    if remaining < df.shape[0]:
                        df = df.iloc[:remaining]
                    collected += df.shape[0]
                    yield df.copy()

    def read_iter(self,
                  from_timestamp: Optional[datetime] = None,
                  to_timestamp: Optional[datetime] = None,
                  columns: Optional[List] = None,
                  selector: Optional[Callable] = None,
                  count: Optional[int] = None,
                  update_timestamp_column: Optional[str] = None) -> Queryable:
        return Queryable(
            self._read_iter(from_timestamp, to_timestamp, columns, selector, count, update_timestamp_column)
        )

    def read(self,
             from_timestamp: Optional[datetime] = None,
             to_timestamp: Optional[datetime] = None,
             columns: Optional[List] = None,
             selector: Optional[Callable] = None,
             count: Optional[int] = None,
             update_timestamp_column: Optional[str] = None,
             download: bool = False
             ):
        if download:
            self.download(from_timestamp, to_timestamp)
        dfs = self.read_iter(from_timestamp, to_timestamp, columns, selector, count, update_timestamp_column).to_list()
        df = pd.concat(dfs, sort=False)
        return df
