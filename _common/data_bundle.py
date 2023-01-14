from typing import *

import os
import pandas as pd
import traceback
import pprint

from pathlib import Path
from yo_fluq_ds import FileIO, Obj, Query
from warnings import warn
import copy
import zipfile
from io import BytesIO

class DataBundle:
    def __init__(self, **frames: pd.DataFrame):
        self.data_frames = {}

        for key, value in frames.items():
            self.data_frames[key] = value

        self.additional_information = Obj()

    def copy(self):
        db =  DataBundle(**self.data_frames)
        db.additional_information = copy.deepcopy(self.additional_information)
        return db

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
            if not isinstance(value, pd.DataFrame):
                result[key] = type(value).__name__
                continue
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
    def _load_zip(fname):
        with zipfile.ZipFile(fname, 'r') as file:
            bundle = DataBundle()
            for f in file.filelist:
                name = f.filename
                parq = '.parquet'
                if name.endswith(parq):
                    buffer = BytesIO(file.read(name))
                    df = pd.read_parquet(buffer)
                    bundle[name.replace(parq, '')] = df
            return bundle

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
        if os.path.isfile(inp):
            return DataBundle._load_zip(inp)
        elif os.path.isdir(inp):
            return DataBundle._read_bundle(inp)
        else:
            raise ValueError(f'{inp} is neither file nor folder')



    def save_as_zip(self, fname):
        with zipfile.ZipFile(fname, 'w', zipfile.ZIP_DEFLATED) as file:
            for name, data in self.data_frames.items():
                bytes = BytesIO()
                data.to_parquet(bytes)
                file.writestr(name + '.parquet', bytes.getbuffer())


    def save(self, folder: Union[str, Path]) -> None:
        if isinstance(folder, str):
            folder = Path(folder)
        os.makedirs(folder, exist_ok=True)
        for key, value in self.data_frames.items():
            value.to_parquet(folder.joinpath(key + '.parquet'))
        FileIO.write_pickle(self.additional_information, folder / 'add_info.pkl')
