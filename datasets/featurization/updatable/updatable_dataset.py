from typing import *

import os
import shutil
import pandas as pd

from datetime import datetime
from pathlib import Path
from yo_fluq_ds import Query
from uuid import uuid4

from ...._common import FileSyncer, Loc
from .._common import PartitionedDatasetRecordHandler, TimePartitionedDatasetBase


class UpdatableDataset(TimePartitionedDatasetBase['UpdatableDataset.DescriptionItem']):
    def __init__(self,
                 location: Union[Path, str],
                 featurizer_name: str,
                 syncer: FileSyncer):
        super(UpdatableDataset, self).__init__(
            location,
            featurizer_name,
            syncer,
            UpdatableDataset.DescriptionHandler
        )

    class DescriptionItem:
        def __init__(self, name: str, timestamp: datetime, is_major: bool, version: str):
            self.name = name
            self.timestamp = timestamp
            self.is_major = is_major
            self.version = version

    DescriptionHandler = PartitionedDatasetRecordHandler('description.parquet', DescriptionItem, 'name')

    def filter_relevant_records(self, records: List[DescriptionItem], from_time: datetime, to_time: datetime, partitions: Optional[List]) -> List[DescriptionItem]:
        if partitions is not None:
            return [z for z in records if z.name in partitions]
        return UpdatableDataset._get_current_records(records, from_time, to_time)

    def get_description_v2(self):
        fname = self.syncer.download_file('description_v2.parquet')
        return pd.read_parquet(fname)


    @staticmethod
    def _get_current_records(
            records: List[DescriptionItem],
            from_timestamp: Optional[datetime],
            to_timestamp: datetime) -> List[DescriptionItem]:
        if to_timestamp is not None:
            records = Query.en(records).where(lambda z: z.timestamp <= to_timestamp).to_list()
        records = Query.en(records).order_by_descending(lambda z: z.timestamp).to_list()
        first_major = Query.en(records).with_indices().where(lambda z: z.value.is_major).select(
            lambda z: z.key).first_or_default()
        if first_major is None:
            raise ValueError(f"There are no major revisions before {to_timestamp}")
        records = records[:first_major + 1]
        if from_timestamp is not None:
            records = [r for r in records if r.timestamp >= from_timestamp]
        return records

    @staticmethod
    def write_to_updatable_dataset(
            syncer: FileSyncer,
            record: DescriptionItem,
            data: Dict[str, pd.DataFrame],
            location: Optional[Union[str, Path]] = None,
    ):
        if location is None:
            location = Loc.temp_path / 'updatable_dataset_updates' / str(uuid4())
            os.makedirs(location)
        else:
            location = Path(location)
            shutil.rmtree(location, ignore_errors=True)
            os.makedirs(location)
        syncer = syncer.change_local_folder(location)
        desc_file = syncer.download_file(UpdatableDataset.DescriptionHandler.get_description_filename())
        if desc_file is not None:
            records = UpdatableDataset.DescriptionHandler.read_parquet(desc_file)
        else:
            records = []
        records.append(record)

        for key, value in data.items():
            file_path = location / record.name / key / 'data.parquet'
            os.makedirs(file_path.parent, exist_ok=True)
            value.to_parquet(file_path)

        syncer.upload_folder(record.name)
        UpdatableDataset.DescriptionHandler.write_parquet(records, location / UpdatableDataset.DescriptionHandler.get_description_filename())
        syncer.upload_file(UpdatableDataset.DescriptionHandler.get_description_filename())
