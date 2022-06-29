from typing import *

import os
import pandas as pd
import traceback
import pprint

from pathlib import Path
from yo_fluq_ds import FileIO, Obj, Query
from warnings import warn


class DataBundle:
    def __init__(self, **frames: pd.DataFrame):
        self.data_frames = {}

        for key, value in frames.items():
            self.data_frames[key] = value

        self.additional_information = Obj()

    def copy(self):
        return DataBundle(**self.data_frames)

    def __getitem__(self, key):
        return self.data_frames[key]

    def __setitem__(self, key, value):
        self.data_frames[key] = value

    def __getattr__(self, key):
        try:
            return self.data_frames[key]
        except KeyError:
            raise AttributeError(key)

    def __contains__(self, item):
        return item in self.data_frames

    def __getstate__(self):
        return (self.data_frames, self.additional_information)

    def __setstate__(self, state):
        self.data_frames, self.additional_information = state

    def describe(self, limits: Optional[int] = None):
        result = {}
        for key, value in self.data_frames.items():
            r = {}
            r['shape'] = value.shape
            r['index_name'] = value.index.name
            if limits is not None:
                cols = list(value.columns[:limits])
                if len(value.columns) > limits:
                    cols.append('...')
                r['columns'] = cols
                rows = list(value.index[:limits])
                if value.shape[0] > limits:
                    rows.append('...')
                r['index'] = rows
            result[key] = r
        return result

    def __str__(self):
        return self.describe().__str__()

    def __repr__(self):
        return self.describe().__str__()

    @staticmethod
    def _read_bundle(path: Path):
        files = (Query
                 .folder(path)
                 .where(lambda z: z.name.endswith('.parquet'))
                 .to_list()
                 )
        data_frames = Query.en(files).to_dictionary(lambda z: z.name.split('.')[0], lambda z: pd.read_parquet(z))
        bundle = DataBundle(**data_frames)

        pkl_fname = str(path / 'add_info.pkl')
        if os.path.exists(pkl_fname):
            add_info = FileIO.read_pickle(pkl_fname)
            bundle.additional_information = add_info

        return bundle

    @staticmethod
    def load(inp: Union[Path, str]):
        """
        Loads bundle from filesystem
        """
        if isinstance(inp, str):
            inp = Path(inp)
        return DataBundle._read_bundle(inp)

    def save(self, folder: Union[str, Path]) -> None:
        if isinstance(folder, str):
            folder = Path(folder)
        os.makedirs(folder, exist_ok=True)
        for key, value in self.data_frames.items():
            value.to_parquet(folder.joinpath(key + '.parquet'))
        FileIO.write_pickle(self.additional_information, folder / 'add_info.pkl')
