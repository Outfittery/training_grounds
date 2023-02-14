from typing import *

import os
import shutil
import uuid

from pathlib import Path

from .featurizer import StreamFeaturizer
from ...._common import Loc, FileSyncer, Logger
from ...access import DataSource, CacheableDataSource


class FeaturizationJob:
    def __init__(self,
                 name: str,
                 version: str,
                 source: DataSource,
                 featurizers: Dict[str, StreamFeaturizer],
                 syncer: Optional[FileSyncer],
                 location: Optional[Union[str, Path]] = None,
                 status_report_frequency: Optional[int] = None,
                 count_of_data_objects: Optional[int] = None
                 ):
        self.name = name
        self.version = version
        self.source = source
        self.featurizers = featurizers

        if location is None:
            self.location = Loc.temp_path / 'featurization_job' / str(uuid.uuid4())
        else:
            self.location = Path(location)

        if syncer is None:
            self.syncer = None
        else:
            self.syncer = syncer.change_local_folder(self.location)

        self.status_report_frequency = status_report_frequency
        self.count_of_data_objects = count_of_data_objects

    def get_name_and_version(self):
        return self.name, self.version

    def _collect(self, df, name):
        df_uid = str(uuid.uuid4())
        file_path = self.location.joinpath(f"{name}/{df_uid}.parquet")
        os.makedirs(file_path.parent, exist_ok=True)
        df.to_parquet(file_path)

    def _send(self):
        if self.syncer is not None:
            self.syncer.upload_folder('')

    def run(self, cache=None):
        Logger.info(f"Featurization Job {self.name} at version {self.version} has started")
        if os.path.exists(self.location):
            shutil.rmtree(self.location)
        os.makedirs(self.location)

        if cache is None:
            flow = self.source.get_data()
        else:
            if isinstance(self.source, CacheableDataSource):
                flow = self.source.safe_cache(cache).get_data()
            else:
                raise ValueError("Cache is requested, but the source is not CacheableDataSource")

        if self.count_of_data_objects is not None:
            flow = flow.take(self.count_of_data_objects)

        for featurizer in self.featurizers.values():
            featurizer.start()

        Logger.info(f"Fetching data")

        self.records_processed_ = 0

        for index, item in enumerate(flow):
            self.records_processed_ += 1
            for name, featurizer in self.featurizers.items():
                df = featurizer.observe_data_point(item)
                if df is not None:
                    self._collect(df, name)
            if self.status_report_frequency is not None and (index + 1) % self.status_report_frequency == 0:
                Logger.info(f"{index+1} data objects are processed")

        Logger.info(f"Data fetched, finalizing")
        for name, featurizer in self.featurizers.items():
            df = featurizer.finish()
            if df is not None:
                self._collect(df, name)

        Logger.info(f"Uploading data")
        self._send()

        Logger.info(f"Featurization job completed")
