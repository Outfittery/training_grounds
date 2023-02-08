from typing import *

from datetime import datetime
from yo_fluq_ds import Query
from dateutil import parser
from pathlib import Path

from .._common import PartitionedDatasetRecordHandler, TimePartitionedDatasetBase
from ...._common import FileSyncer


class DayPartitionedDataset(TimePartitionedDatasetBase['PartitionedDataset.DescriptionItem']):
    def __init__(self,
                 location: Union[Path, str],
                 featurizer_name: str,
                 syncer: FileSyncer):
        super(DayPartitionedDataset, self).__init__(
            location,
            featurizer_name,
            syncer,
            DayPartitionedDataset.DescriptionHandler
        )

    class DescriptionItem:
        def __init__(self, name, timestamp, version):
            self.name = name
            self.timestamp = timestamp
            self.version = version

    DescriptionHandler = PartitionedDatasetRecordHandler('partitions_description.parquet', DescriptionItem, 'name')

    def filter_relevant_records(self, records: List[DescriptionItem], from_time: datetime, to_time: datetime, partitions) -> List[DescriptionItem]:
        records = Query.en(records)
        if to_time is not None:
            records = records.where(lambda z: parser.parse(z.name) < to_time)
        if from_time is not None:
            records = records.where(lambda z: parser.parse(z.name) >= from_time)
        records = records.distinct(lambda z: z.name)
        records = records.to_list()
        return records
