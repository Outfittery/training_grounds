from .._common import S3Handler, DataBundle, Logger
from .accessor import IAccessor
from .cache import Cache
from pathlib import Path
import os
import shutil
import pandas as pd
from yo_fluq_ds import FileIO



class S3AccessorInstance(IAccessor):
    def __init__(self, s3_bucket, s3_path, method, method_name, opener, opener_name):
        self.s3_bucket = s3_bucket
        self.s3_path = s3_path
        self.method = method
        self.method_name = method_name
        self.opener_name = opener_name
        self.opener = opener

    def get_name(self) -> str:
        return Cache.default_name('s3',self.s3_bucket, self.s3_path, self.method_name, self.opener_name)

    def get_data(self):
        Logger.info(f'Accessing {self.s3_bucket}//{self.s3_path} by reading method `{self.method_name}`')
        path = Path(self.method(self.s3_bucket, self.s3_path))
        Logger.info(f'Opening local file {path} with opener `{self.opener_name}`')
        data = self.opener(path)
        if path.is_file():
            os.unlink(path)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        return data

class S3AccessMethod:
    def __init__(self, s3_bucket, s3_path, method, method_name):
        self.s3_bucket = s3_bucket
        self.s3_path = s3_path
        self.method = method
        self.method_name = method_name

    def open_parquet(self) -> Cache:
        return Cache.source(S3AccessorInstance(self.s3_bucket, self.s3_path, self.method, self.method_name, pd.read_parquet, 'parquet'))

    def open_bundle(self) -> Cache:
        return Cache.source(S3AccessorInstance(self.s3_bucket, self.s3_path, self.method, self.method_name, DataBundle.load, 'bundle'))

    def open_pickle(self) -> Cache:
        return Cache.source(S3AccessorInstance(self.s3_bucket, self.s3_path, self.method, self.method_name, FileIO.read_pickle, 'pickle'))


def max_file(s3_bucket, s3_path):
    if not s3_path.endswith('/'):
        raise ValueError('For maxfile, the path must be a folder, and it must end with `/`')
    last = max(S3Handler.get_folder_content(s3_bucket, s3_path))
    path = S3Handler.download_file(s3_bucket, last)
    return path


class S3Accessor:
    def __init__(self, s3_bucket, s3_path):
        self.s3_bucket = s3_bucket
        self.s3_path = s3_path

    def folder(self):
        return S3AccessMethod(self.s3_bucket, self.s3_path, S3Handler.download_folder, 'folder')

    def file(self):
        return S3AccessMethod(self.s3_bucket, self.s3_path, S3Handler.download_file, 'file')

    def max_file(self):
        return S3AccessMethod(self.s3_bucket, self.s3_path, max_file, 'max_file')












