from typing import *
import pickle
import shutil

from pathlib import Path

from .arch import AbstractCacheDataSource

from yo_fluq_ds import agg, Queryable, FileIO, Query

import os



class AbstractFileDataSource(AbstractCacheDataSource):
    """
    The abstraction for getting the data from the file cache.
    Inheritants of this class represent various formats of file.
    """

    def __init__(self, path, extension):
        path = str(path)
        self.path = path + "." + extension
        self.length_path = path + '.' + extension + ".length"

    def _self_get_reading_query(self):
        raise NotImplementedError()

    def get_data(self):
        try:
            length = int(FileIO.read_text(self.length_path))
        except:
            length = None
        q = self._self_get_reading_query()
        return Queryable(q, length)

    def _get_writing_aggregator(self, tmp_path):
        raise NotImplementedError()

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
        file_agg = self._get_writing_aggregator(tmp_path)
        pipeline = Query.push().split_pipelines(
            file=file_agg,
            cnt=agg.Count()
        )
        result = q.feed(pipeline)
        if os.path.isfile(full_path):
            os.remove(full_path)
        shutil.move(tmp_path, full_path)
        FileIO.write_text(str(result['cnt']), self.length_path)

    def is_available(self):
        return os.path.isfile(self.path)


import zipfile


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

    def on_enter(self, instance):
        if self.replace and os.path.isfile(self.filename):
            os.remove(self.filename)
        instance.file = zipfile.ZipFile(self.filename, 'a', compression=self.compression)
        instance.buffer = []
        instance.counter = 0

    def _flush(self, instance):
        instance.file.writestr(str(instance.counter), self.writer(instance.buffer))
        instance.buffer = []
        instance.counter += 1

    def on_process(self, instance, element):
        instance.buffer.append(element)
        if len(instance.buffer) >= self.buffer_size:
            self._flush(instance)

    def on_report(self, instance):
        return None

    def on_exit(self, instance, exc_type, exc_val, exc_tb):
        self._flush(instance)
        instance.file.__exit__(exc_type, exc_val, exc_tb)


class ZippedFileDataSource(AbstractFileDataSource):
    """
    File cache in .zip format. Records are placed in files (1000 records in each), then files are zipped.
    This format is not memory-extensive, and have a stable and low reading time
    Buffering is important: when opening zip-file, it's table of contents is read, so if we have 1 file per record,
    opening a file takes a long time and consumes a lot of memory
    """

    def __init__(self, path, buffer_size=1000):
        super(ZippedFileDataSource, self).__init__(path, 'pkllines.zip')
        self.buffer_size = buffer_size

    def _get_writing_aggregator(self, tmp_path):
        return _ToZipFolderBuffered(tmp_path, buffer_size=self.buffer_size)

    def _self_get_reading_query(self):
        return Query.file.zipped_folder(self.path).select_many(lambda z: z.value)
