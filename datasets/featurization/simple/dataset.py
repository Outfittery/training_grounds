from typing import *

import os
import pandas as pd

from pathlib import Path
from yo_fluq_ds import Query, Queryable
from ...access import LambdaDataFrameSource
from ...._common import FileSyncer
import shutil

class Dataset:
    def __init__(self, dataset_location: Union[str,Path], syncer: Optional[FileSyncer]):
        self.dataset_location = Path(dataset_location)
        if syncer is not None:
            self.syncer = syncer.change_local_folder(self.dataset_location)
        else:
            self.syncer = None


    def is_available(self):
        return os.path.isdir(self.dataset_location)


    def download(self):
        if self.syncer is not None:
            shutil.rmtree(self.dataset_location, ignore_errors=True)
            self.syncer.download_folder('')


    def _open_file(self, file, columns, selector):
        df = pd.read_parquet(file, columns=columns)
        if selector is not None:
            df = selector(df)
        return df


    def _read_iter(self, files, columns, selector, count):
        collected = 0
        for file in files:
            df = self._open_file(file, columns, selector)
            if count is None:
                yield df
            else:
                remaining = count - collected
                if remaining <= 0:
                    break
                if remaining < df.shape[0]:
                    df = df.iloc[:remaining]
                collected+=df.shape[0]
                yield df


    def read_iter(self,
                  columns:Optional[List] = None,
                  selector: Optional[Callable[[pd.DataFrame],pd.DataFrame]] = None,
                  count: Optional[int] = None
                  ):
        files = Query.folder(self.dataset_location).order_by(lambda z: z.name).to_list()
        return Queryable(
            self._read_iter(files, columns, selector, count),
            len(files)
        )



    def read(self,
             columns:Optional[List] = None,
             selector: Optional[Callable[[pd.DataFrame],pd.DataFrame]] = None,
             count: Optional[int] = None,
             download: bool = False
             ):
        if download:
            self.download()
        dfs = self.read_iter(columns,selector,count).to_list()
        df = pd.concat(dfs,sort=False)
        return df


    def as_data_frame_source(self, **kwargs):
        kwargs['download'] = True
        return LambdaDataFrameSource(self.read, **kwargs)

