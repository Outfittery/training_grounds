from typing import *

import os
import logging
import shutil
import uuid

from .featurizer import StreamFeaturizer
from ..._common import Loc
from ..access import DataSource, CacheableDataSource



logger = logging.getLogger()


class FeaturizationJobDestination:
    def setup(self):
        raise NotImplementedError()

    def collect(self, df, name):
        raise NotImplementedError()

    def send(self):
        raise NotImplementedError()


class LocalFileJobDestination(FeaturizationJobDestination):
    def __init__(self,
                 location
                 ):
        self.location = Loc.temp_path.joinpath(location)
        self.names = []

    def setup(self):
        if os.path.exists(self.location):
            shutil.rmtree(self.location)

    def collect(self, df, name):
        if name not in self.names:
            self.names.append(name)
        df_uid = str(uuid.uuid4())
        file_path = self.location.joinpath(f"{name}/{df_uid}.parquet")
        os.makedirs(file_path.parent, exist_ok=True)
        df.to_parquet(file_path)

    def send(self):
        pass


class InMemoryJobDestination(FeaturizationJobDestination):
    def __init__(self):
        self.buffer = None

    def setup(self):
        self.buffer = {}

    def collect(self, df, name):
        if name not in self.buffer:
            self.buffer[name]=[]
        self.buffer[name].append(df)

    def send(self):
        pass



class FeaturizationJob:
    def __init__(self,
                 name: str,
                 version: str,
                 source: DataSource,
                 featurizers: Dict[str,StreamFeaturizer],
                 destination: FeaturizationJobDestination,
                 status_report_frequency: Optional[int] = None,
                 count_of_data_objects: Optional[int] = None
                 ):
        self.name = name
        self.version = version
        self.source = source
        self.featurizers = featurizers
        self.destination = destination
        self.status_report_frequency = status_report_frequency
        self.count_of_data_objects = count_of_data_objects

    def get_name_and_version(self):
        return self.name, self.version

    def run(self, cache = None):
        logger.info(f"Featurization Job {self.name} at version {self.version} has started")
        self.destination.setup()


        if cache is None:
            flow = self.source.get_data()
        else:
            if isinstance(self.source,CacheableDataSource):
                flow = self.source.safe_cache(cache).get_data()
            else:
                raise ValueError("Cache is requested, but the source is not CacheableDataSource")

        if self.count_of_data_objects is not None:
            flow = flow.take(self.count_of_data_objects)

        for featurizer in self.featurizers.values():
            featurizer.start()

        logger.info(f"Fetching data")

        for index, item in enumerate(flow):
            for name, featurizer in self.featurizers.items():
                df = featurizer.observe_data_point(item)
                if df is not None:
                    self.destination.collect(df, name)
            if self.status_report_frequency is not None and (index+1) % self.status_report_frequency == 0:
                logger.info(f"{index+1} data objects are processed")

        logger.info(f"Data fetched, finalizing")
        for name, featurizer in self.featurizers.items():
            df = featurizer.finish()
            if df is not None:
                self.destination.collect(df, name)

        logger.info(f"Uploading data")
        self.destination.send()

        logger.info(f"Featurization job completed")
