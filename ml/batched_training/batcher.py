from typing import *

import pandas as pd

from .batcher_strategy import BatcherStrategy, SimpleBatcherStrategy
from .data_bundle import DataBundle, IndexedDataBundle
from .extractors import Extractor



class Batcher:
    """
    Entity that produces batches out of the bundle.

    For each dataframe in ``bundle.data_frames``, it contains a transformer (e.g. ``DataFrameTransformer``) that converts the parts of the dataframe
    This is done here to reduce memory consumption: e.g., in case of OneHotEncoding, transforming creates lots of additional columns, and it's better to do it
    per batch, not per the whole dataset.

    If there is no transformer per data_frame, the corresponding data will not be included in batches
    """

    def __init__(self,
                 batch_size: int,
                 extractors: List[Extractor],
                 batching_strategy: Optional[BatcherStrategy] = None
                 ):
        self.batch_size = batch_size
        self.extractors = extractors
        self.default_batching_strategy = SimpleBatcherStrategy()
        self.batching_strategy = batching_strategy or SimpleBatcherStrategy()

    def fit_extractors(self, bundle: DataBundle) -> None:
        for extractor in self.extractors:
            extractor.fit(bundle)

    def _get_strategy(self, force_default_strategy):
        if force_default_strategy:
            return self.default_batching_strategy
        else:
            return self.batching_strategy

    def _get_index(self, db: IndexedDataBundle):
        return db.bundle.index_frame.loc[db.index]

    def get_batch_count(self, db: IndexedDataBundle, force_default_strategy=False) -> int:
        return self._get_strategy(force_default_strategy).get_batch_count(self.batch_size, self._get_index(db))

    def get_batch(self, db: IndexedDataBundle, batch_index: int, force_default_strategy=False) -> Dict[str, pd.DataFrame]:
        """
        Args:
            db:
            batch_index:

        Returns: a dictionary with batch component, one per key of ``self.transformers``
        """
        index = self._get_strategy(force_default_strategy).get_batch(self.batch_size, self._get_index(db), batch_index)
        index_df = db.bundle.index_frame.loc[index]
        batch = Extractor.make_extraction(index_df, db.bundle, self.extractors)
        return batch
