from typing import *
from ..._common import Loc, S3Handler, DataBundle, Logger
import pandas as pd
import sqlalchemy as db
from yo_fluq_ds import Query, fluq
import uuid


class S3BundleSource:
    def __init__(self, s3_bucket, s3_path, name = None):
        self.s3_bucket = s3_bucket
        self.s3_path = s3_path
        self.name = name
        if self.name is None:
            self.name = str(uuid.uuid4())

    def get_bundle(self):
        Logger.info(f'Downloading bundle from {self.s3_bucket}//{self.s3_path}')
        path= Loc.temp_path/self.name
        S3Handler.download_folder(self.s3_bucket, self.s3_path, path)
        bundle = DataBundle.load(path)
        Logger.info('Done')
        return bundle


class LocalBundleSource:
    def __init__(self, path):
        self.path = path

    def get_bundle(self):
        Logger.info(f'Loading file from {self.path}')
        bundle = DataBundle.load(self.path)
        Logger.info('Done')
        return bundle
