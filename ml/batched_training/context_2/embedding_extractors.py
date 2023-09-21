from typing import *
from ..extractors import Extractor, IndexedDataBundle
import pandas as pd
import numpy as np

class AbstractEmbeddingExtractor(Extractor):
    def __init__(self,
                 inner_extractor: Extractor,
                 unk_name: str,
                 ):
        self.inner_extractor = inner_extractor
        self.mapping_ = None
        self.unk_name = unk_name

    def extract_inner(self, ibundle):
        values = self.inner_extractor.extract(ibundle)
        if len(values.columns) != 1:
            raise ValueError(
                f'Embedding extractor expects exactly one column from the extractor, but there were several:\n{list(values.columns)}')
        return values[values.columns[0]]

    def extract(self, ibundle: IndexedDataBundle) -> pd.DataFrame:
        s = self.extract_inner(ibundle)
        s = pd.Series(np.where(s.isin(self.mapping_), s, self.unk_name), index=s.index)
        s = s.map(self.mapping_)
        return pd.DataFrame({self.inner_extractor.get_name(): s}, index=s.index)

    def get_name(self):
        return self.inner_extractor.get_name()


class NewEmbeddingExtractor(AbstractEmbeddingExtractor):
    def __init__(self,
                 inner_extractor: Extractor,
                 vocab_size: int,
                 unk_name: str = '<unk>',
                 ):
        super().__init__(inner_extractor, unk_name)
        self.mapping_ = None
        self.vocab_size = vocab_size

    def fit(self, ibundle: IndexedDataBundle):
        self.inner_extractor.fit(ibundle)
        values = self.extract_inner(ibundle)
        sizes = values.to_frame('v').groupby('v').size().sort_values(ascending=False)
        most_popular_values = list(sizes.loc[sizes.index != self.unk_name].index)[:self.vocab_size - 1]
        self.mapping_ = {v: i + 1 for i, v in enumerate(most_popular_values)}
        self.mapping_[self.unk_name] = 0


class ExistingEmbeddingExtractor(AbstractEmbeddingExtractor):
    def __init__(self,
                 inner_extractor: Extractor,
                 vectors: pd.DataFrame,
                 unk_name: str,
                 ):
        super().__init__(inner_extractor, unk_name)
        self.mapping_ = {k:v for v, k in enumerate(vectors.index)}

    def fit(self, ibundle: IndexedDataBundle):
        pass
        


