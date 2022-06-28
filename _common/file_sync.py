from typing import *
import boto3
import botocore

import os
import io
import pandas as pd

from pathlib import Path

from .s3helpers import S3Handler
from yo_fluq_ds import FileIO, Query


class FileSyncer:
    def get_local_folder(self) -> Path:
        raise NotImplementedError()

    def get_remote_subfolder(self) -> str:
        raise  NotImplementedError()

    def download_file(self, subpath: str) -> Optional[Path]:
        raise NotImplementedError()

    def download_folder(self, subpath: str) -> Optional[Path]:
        raise NotImplementedError()

    def upload_file(self, subpath: str):
        raise NotImplementedError()

    def upload_folder(self, subpath):
        raise NotImplementedError()

    def cd(self, subpath: str) -> 'FileSyncer':
        return (self
                .change_local_folder(self.get_local_folder()/subpath)
                .change_remote_subfolder(os.path.join(self.get_remote_subfolder(), subpath))
        )

    def change_local_folder(self, path: Path) -> 'FileSyncer':
        raise NotImplementedError()

    def change_remote_subfolder(self, subpath: str) -> 'FileSyncer':
        raise NotImplementedError()




class S3FileSyncer(FileSyncer):
    def __init__(self, bucket:str, prefix:str, file_location: Optional[Path] = None):
        self.bucket = bucket
        self.prefix = prefix
        self.file_location = file_location

    def get_local_folder(self):
        return self.file_location

    def get_remote_subfolder(self) -> str:
        return self.prefix

    def download_file(self, subpath: str) -> Optional[Path]:
        location = self.file_location / subpath
        os.makedirs(str(location.parent), exist_ok=True)
        try:
            S3Handler.download_file(self.bucket, os.path.join(self.prefix, subpath), location)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return None
            else:
                raise
        return location


    def download_folder(self, subpath: str) -> Optional[Path]:
        return self.download_folder_with_reporting(subpath, None)

    def download_folder_with_reporting(self, subpath, reporting) -> Optional[Path]:
        location = self.file_location / subpath
        S3Handler.download_folder(self.bucket,
                                  os.path.join(self.prefix, subpath),
                                  location,
                                  reporting
                                  )
        return location

    def upload_file(self, subpath: str):
        S3Handler.upload_file(self.bucket, os.path.join(self.prefix, subpath), self.file_location/subpath)

    def upload_folder(self, subpath):
        S3Handler.upload_folder(self.bucket, os.path.join(self.prefix, subpath), self.file_location/subpath)


    def change_local_folder(self, path: Path):
        return S3FileSyncer(self.bucket, self.prefix, path)

    def change_remote_subfolder(self, subpath: str) -> 'FileSyncer':
        return S3FileSyncer(self.bucket, subpath, self.file_location)



class MemoryFileSyncer(FileSyncer):
    def __init__(self, root = None, prefix='', cache = None):
        self.root = root
        self.prefix = prefix
        if cache is None:
            self.cache = OrderedDict()
        else:
            self.cache = cache

    def get_local_folder(self) -> Path:
        return self.root

    def get_remote_subfolder(self) -> str:
        return self.prefix

    def get_file_stream(self, key: Union[int, str, Callable]):
        if isinstance(key, int):
            key = list(self.cache)[key]
        elif isinstance(key, Callable):
            key = Query.en(self.cache).where(key).first()
        return io.BytesIO(self.cache[key])

    def get_parquet(self, key: Union[int, str, Callable]):
        return pd.read_parquet(self.get_file_stream(key))



    def get_sub(self, subpath):
        return os.path.join(self.prefix,subpath)

    def get_trim(self, s):
        if self.prefix=='':
            return s
        return s[len(self.prefix)+1:]

    def download_file(self, subpath: str) -> Optional[Path]:
        path = self.root/subpath
        os.makedirs(str(path.parent), exist_ok=True)
        key = self.get_sub(subpath)
        if key not in self.cache:
            return None
        with open(path, 'wb') as f:
            f.write(self.cache[key])
        return path

    def download_folder(self, subpath: str) -> Optional[Path]:
        path = self.root/subpath
        os.makedirs(str(path), exist_ok=True)
        for key, value in self.cache.items():
            if key.startswith(self.get_sub(subpath)):
                self.download_file(self.get_trim(key))
        return path

    def upload_file(self, subpath: str):
        with open(self.root/subpath,'rb') as f:
            self.cache[self.get_sub(subpath)]=f.read()

    def upload_folder(self, subpath):
        for key in list(self.cache):
            if key.startswith(self.get_sub(subpath)):
                del self.cache[key]

        for path in Query.folder(self.root/subpath,'**/*'):
            if path.is_file():
                self.upload_file(path.relative_to(self.root))

    def change_local_folder(self, path: Path) -> 'FileSyncer':
        return MemoryFileSyncer(path,self.prefix, self.cache)

    def change_remote_subfolder(self, subpath: str) -> 'FileSyncer':
        return MemoryFileSyncer(self.root, subpath, self.cache)



