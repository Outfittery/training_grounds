from typing import *

import os
import pandas as pd

from pathlib import Path
from yo_fluq_ds import Query



class DataBundle:
    """
    Keeps the data, required by batched training.

    Consider data frames and index that describe how to merge them.
    """
    def __init__(self,
                 index_frame: pd.DataFrame,
                 data_frames: Dict[str, pd.DataFrame]
                 ):
        self.index_frame = index_frame
        self.data_frames = data_frames

    @staticmethod
    def _read_bundle(path: Path):
        index_frame = pd.read_parquet(path.joinpath('index.parquet'))
        files = (Query
                 .folder(path)
                 .where(lambda z: z.name!='index.parquet')
                 .where(lambda z: z.name.endswith('.parquet'))
                 .to_list()
                 )
        data_frames = Query.en(files).to_dictionary(lambda z: z.name.split('.')[0], lambda z: pd.read_parquet(z))
        return DataBundle(
            index_frame,
            data_frames
        )

    @staticmethod
    def ensure(inp: Union[Path,str,'DataBundle'])->'DataBundle':
        """
        Converts bundle or path to it into a bundle
        Args:
            inp:

        Returns:

        """
        if isinstance(inp,DataBundle):
            return inp
        if isinstance(inp, str):
            inp = Path(inp)
        if os.path.isdir(inp):
            return DataBundle._read_bundle(inp)
        else:
            raise ValueError(f'{inp} is neither DataBundle not the path to folder that contains Data Bundle parquet files')

    @staticmethod
    def load(inp: Union[Path,str]):
        """
        Loads bundle from filesystem
        """
        if isinstance(inp,str):
            inp = Path(inp)
        return DataBundle._read_bundle(inp)

    def save(self, folder: Union[str,Path]) -> None:
        if isinstance(folder,str):
            folder = Path(folder)
        os.makedirs(folder, exist_ok=True)
        self.index_frame.to_parquet(folder.joinpath('index.parquet'))
        for key, value in self.data_frames.items():
            value.to_parquet(folder.joinpath(key+'.parquet'))

    def as_indexed(self, custom_index = None) -> 'IndexedDataBundle':
        if custom_index is None:
            custom_index = self.index_frame.index
        return IndexedDataBundle(self, custom_index)



class IndexedDataBundle:
    def __init__(self, bundle: DataBundle, index: pd.Index):
        self.bundle = bundle
        self.index = index
