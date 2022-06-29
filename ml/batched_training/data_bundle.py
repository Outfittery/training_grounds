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
