from typing import *

import pandas as pd

from .extractors import Extractor, DataBundle
from .plain_extractor import PlainExtractor
from .batcher import Batcher, IndexedDataBundle


class PrecomputingExtractor(Extractor):
    def __init__(self,
                 name: str,
                 inner_extractor: Extractor,
                 index_size_for_precomputing: Optional[int] = None,
                 index_filter_for_precomputing: Optional[Callable[[pd.DataFrame], pd.Index]] = None,
                 ):
        self.name = name
        self.inner_extractor = inner_extractor
        self.fitted = False
        self.index_size_for_precomputing = index_size_for_precomputing
        self.index_filter_for_precomputing = index_filter_for_precomputing
        self.extractor_from_preprocessed = PlainExtractor.build(name).index(name).apply()

    def _inner_fit(self, ibundle: IndexedDataBundle):
        self.inner_extractor.fit(ibundle)
        self.fitted = True

    def _precompute(self, ibundle: IndexedDataBundle):
        if self.index_filter_for_precomputing is not None:
            index = self.index_filter_for_precomputing(ibundle.index_frame)
        else:
            index = ibundle.index_frame.index
        if self.index_size_for_precomputing is not None:
            batch_size = self.index_size_for_precomputing
        else:
            batch_size = len(index)
        ibundle = ibundle.change_index(index)
        batcher = Batcher([self.inner_extractor])
        dfs = []
        key = None
        for i in range(batcher.get_batch_count(batch_size, ibundle)):
            batch = batcher.get_batch(batch_size, ibundle, i)
            if key is None:
                key = [c for c in batch.bundle.data_frames.keys() if c != 'index'][0]
            dfs.append(batch[key])
        return dfs

    def _fill_bundle(self, ibundle: IndexedDataBundle):
        dfs = self._precompute(ibundle)
        if len(dfs) > 0:
            df = pd.concat(dfs, axis=0)
        else:
            df = pd.DataFrame([])
        ibundle.bundle[self.name] = df

    def preprocess_bundle(self, ibundle: IndexedDataBundle):
        if self.fitted:
            self._fill_bundle(ibundle)

    def fit(self, ibundle: IndexedDataBundle):
        self._inner_fit(ibundle)
        self._fill_bundle(ibundle)

    def extract(self, ibundle: IndexedDataBundle) -> pd.DataFrame:
        return self.extractor_from_preprocessed.extract(ibundle)
