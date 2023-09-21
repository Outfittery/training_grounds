from typing import *

import pandas as pd

from ..._common import DataBundle


class IndexedDataBundle:
    def __init__(self, index_frame: pd.DataFrame, bundle: DataBundle):
        self.bundle = bundle
        self.index_frame = index_frame

    def change_index(self, index: Union[pd.Index, pd.DataFrame]):
        if isinstance(index, pd.Index):
            return IndexedDataBundle(self.index_frame.loc[index], self.bundle)
        elif isinstance(index, pd.DataFrame):
            return IndexedDataBundle(index, self.bundle)
        else:
            raise ValueError(f'`index` is expected to be pd.Index or pd.Dataframe, but was {type(index)}')

    def __getitem__(self, key):
        if key=='index':
            return self.index_frame
        return self.bundle.data_frames[key]

    def describe(self, limits: Optional[int] = None):
        desc = self.bundle.describe(limits)
        desc['@index_frame'] = self.bundle.describe_dataframe(self.index_frame, limits)
        return desc

    def __repr__(self):
        return self.describe().__str__()