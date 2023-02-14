from typing import *

import os
import shutil

from pathlib import Path
from datetime import datetime
from uuid import uuid4
from yo_fluq_ds import Query

from ...._common import FileSyncer, Loc
from .updatable_dataset import UpdatableDataset


class UpdatableDatasetScoringInstance:
    def __init__(self,
                 job: 'UpdatableDatasetScoringJob',
                 uid: str,
                 dst_featurizer: str,
                 src_featurizer: str,
                 src_syncer: FileSyncer,
                 method: Callable,
                 start_from_time: Optional[datetime],
                 current_time: datetime,
                 force_full_update: bool,
                 ):
        self.job = job
        self.uid = uid
        self.src_featurizer = src_featurizer
        self.src_syncer = src_syncer
        self.dst_featurizer = dst_featurizer
        self.method = method
        self.start_from_time = start_from_time
        self.current_time = current_time
        self.force_full_update = force_full_update
        self.is_major_ = None  # type: Optional[bool]

    def run(self):
        src_location = self.job.src_folder / self.src_featurizer
        dst_location = self.job.dst_folder / self.uid / self.dst_featurizer
        os.makedirs(str(dst_location))

        dataset = UpdatableDataset(
            src_location,
            self.src_featurizer,
            self.src_syncer
        )
        downloaded_records = dataset.download(self.start_from_time, self.current_time)
        for df in dataset.read_iter(
            from_timestamp=self.start_from_time,
            to_timestamp=self.current_time,

        ):
            rdf = self.method(df)  # type: pd.DataFrame
            rdf.to_parquet(dst_location / (str(uuid4()) + '.parquet'))

        self.job.dst_syncer.upload_folder(os.path.join(self.uid, self.dst_featurizer))

        self.is_major_ = Query.en(downloaded_records).any(lambda z: z.is_major)


class UpdatableDatasetScoringMethod:
    def __init__(self,
                 dst_featurizer: str,
                 src_syncer: FileSyncer,
                 src_featurizer: str,
                 method: Callable):
        self.dst_featurizer = dst_featurizer
        self.src_syncer = src_syncer
        self.src_featurizer = src_featurizer
        self.method = method


class UpdatableDatasetScoringJob:
    def __init__(self,
                 name: str,
                 version: str,
                 dst_syncer: FileSyncer,
                 methods: List[UpdatableDatasetScoringMethod],
                 location: Optional[Union[str, Path]] = None,
                 ):
        self.name = name
        self.version = version
        self.methods = methods
        self.records = None  # type: Optional[List[UpdatableDataset.DescriptionItem]]

        if location is None:
            self.location = Loc.temp_path / 'updatable_dataset_scoring_job' / str(uuid4())
        else:
            self.location = Path(location)

        self.src_folder = self.location / 'src'
        self.dst_folder = self.location / 'dst'
        self.dst_syncer = dst_syncer.change_local_folder(self.dst_folder)

    def get_name_and_version(self):
        return self.name, self.version

    def _initiate(self):
        shutil.rmtree(self.location, ignore_errors=True)
        os.makedirs(str(self.src_folder))
        os.makedirs(str(self.dst_folder))
        path = self.dst_syncer.download_file(UpdatableDataset.DescriptionHandler.get_description_filename())
        if path is None:
            self.records = []
            return None
        self.records = UpdatableDataset.DescriptionHandler.read_parquet(path)
        return self.records[-1].timestamp

    def run(self,
            current_time: Optional[datetime] = None,
            force_full_update: bool = False,
            custom_revision_id: Optional[str] = None,
            custom_start_time: Optional[datetime] = None
            ):

        start_from = self._initiate()

        if force_full_update:
            start_from = None
        elif custom_start_time is not None:
            start_from = custom_start_time

        if custom_revision_id is None:
            uid = str(uuid4())
        else:
            uid = custom_revision_id

        if current_time is None:
            current_time = datetime.now()

        is_major = True

        for method in self.methods:
            instance = UpdatableDatasetScoringInstance(
                self,
                uid,
                method.dst_featurizer,
                method.src_featurizer,
                method.src_syncer,
                method.method,
                start_from,
                current_time,
                force_full_update
            )
            instance.run()
            is_major = is_major and instance.is_major_

        self.records.append(UpdatableDataset.DescriptionItem(
            uid,
            current_time,
            is_major,
            self.version
        ))
        UpdatableDataset.DescriptionHandler.write_parquet_to_folder(self.records, self.dst_folder)
        self.dst_syncer.upload_file(UpdatableDataset.DescriptionHandler.get_description_filename())
