import os

from pathlib import Path

from ..._common import S3Handler



class AbstractDatasetDownloader:
    def download_file(self, common_location: Path, subpath: str):
        raise NotImplementedError()

    def download_folder(self, common_location: Path, subpath: str, reporting: str):
        raise NotImplementedError()


class S3DatasetDownloader(AbstractDatasetDownloader):
    def __init__(self, bucket, prefix):
        self.bucket = bucket
        self.prefix = prefix

    def download_file(self, common_location: Path, subpath: str):
        location = common_location / subpath
        os.makedirs(str(location.parent), exist_ok=True)
        S3Handler.download_file(self.bucket, os.path.join(self.prefix, subpath), location)

    def download_folder(self, common_location: Path, subpath: str, reporting: str):
        S3Handler.download_folder(self.bucket, os.path.join(self.prefix, subpath), common_location / subpath, reporting)
