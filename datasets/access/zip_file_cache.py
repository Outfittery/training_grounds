from typing import *

import pickle
import shutil
import os

from pathlib import Path
from yo_fluq_ds import agg, Queryable, FileIO, Query

from .arch import AbstractCacheDataSource
import zipfile


class ZippedFileDataSource(AbstractCacheDataSource):
    """
    File cache in .zip format. Records are placed in files (1000 records in each), then files are zipped.
    This format is not memory-extensive, and have a stable and low reading time
    Buffering is important: when opening zip-file, it's table of contents is read, so if we have 1 file per record,
    opening a file takes a long time and consumes a lot of memory
    """

    def __init__(self, path, buffer_size=1000):
        path = str(path)
        self.path = path
        self.buffer_size = buffer_size

    def cache_from(self, src: Queryable, cnt=None) -> None:
        """
        Caches data from a given queryable (for instance, from one produced by DataSource::get_data).
        Args:
            src: Queryable to cache from
            cnt: amount of objects to cache

        Returns:

        """
        q = src
        if cnt is not None:
            q = q.take(cnt)
        full_path = str(self.path)
        os.makedirs(Path(full_path).parent.__str__(), exist_ok=True)
        tmp_path = full_path + '.tmp'
        file_agg = _ToZipFolderBuffered(tmp_path, buffer_size=self.buffer_size)
        q.feed(file_agg)
        if os.path.isfile(full_path):
            os.remove(full_path)
        shutil.move(tmp_path, full_path)

    def is_available(self):
        return os.path.isfile(self.path)

    def _get_data_iter(self, prefix: str):
        with zipfile.ZipFile(self.path, 'r') as zfile:
            for name in zfile.namelist():
                if not name.startswith(prefix):
                    continue
                for element in pickle.loads(zfile.read(name)):
                    yield element

    def get_data(self) -> Queryable:
        length = None
        prefix = ''
        with zipfile.ZipFile(self.path, 'r') as zfile:
            if 'length' in zfile.namelist():
                length = int(zfile.read('length').decode('utf-8'))
                prefix = 'data'

        return Queryable(self._get_data_iter(prefix), length)





class _ToZipFolderBuffered(agg.PushQueryElement):
    def __init__(self,
                 filename: Union[str, Path],
                 writer: Callable = pickle.dumps,
                 replace=True, compression=zipfile.ZIP_DEFLATED,
                 buffer_size=1000):
        self.filename = filename
        self.compression = compression
        self.writer = writer
        self.replace = replace
        self.buffer_size = buffer_size
        self.length = 0

    def on_enter(self, instance):
        if self.replace and os.path.isfile(self.filename):
            os.remove(self.filename)
        instance.file = zipfile.ZipFile(self.filename, 'a', compression=self.compression)
        instance.buffer = []
        instance.counter = 0

    def _flush(self, instance):
        instance.file.writestr(f'data/{instance.counter}', self.writer(instance.buffer))
        instance.buffer = []
        instance.counter += 1

    def on_process(self, instance, element):
        instance.buffer.append(element)
        if len(instance.buffer) >= self.buffer_size:
            self._flush(instance)
        self.length += 1

    def on_report(self, instance):
        return None

    def on_exit(self, instance, exc_type, exc_val, exc_tb):
        self._flush(instance)
        instance.file.writestr('length', str(self.length).encode('utf-8'))
        instance.file.__exit__(exc_type, exc_val, exc_tb)
