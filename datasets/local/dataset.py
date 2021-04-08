from typing import *

import os
import pandas as pd

from pathlib import Path
from yo_fluq_ds import Query, Queryable

from .downloader import AbstractDatasetDownloader



class Dataset:
    def __init__(self, common_location: Union[str,Path], subpath:str, downloader: Optional[AbstractDatasetDownloader]=None):
        self.common_location = Path(common_location)
        self.subpath = subpath
        self.downloader = downloader
        self._folder_location =self.common_location/self.subpath


    def is_available(self):
        return os.path.isdir(self._folder_location)


    def download(self, reporting=None):
        if self.downloader is not None:
            self.downloader.download_folder(self.common_location, self.subpath, reporting)


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
        files = Query.folder(self._folder_location).order_by(lambda z: z.name).to_list()
        return Queryable(
            self._read_iter(files, columns, selector, count),
            len(files)
        )



    def read(self,
             columns:Optional[List] = None,
             selector: Optional[Callable[[pd.DataFrame],pd.DataFrame]] = None,
             count: Optional[int] = None):
        dfs = self.read_iter(columns,selector,count).to_list()
        df = pd.concat(dfs,sort=False)
        return df

