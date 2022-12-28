from typing import *

import pandas as pd

from .batcher_strategy import BatcherStrategy, SimpleBatcherStrategy
from .data_bundle import DataBundle, IndexedDataBundle
from .extractors import Extractor


#TODO: Why is batch a  dictionary and not a bundle?

class Batcher:
    """
    Entity that produces batches out of the bundle.

    For each dataframe in ``bundle.data_frames``, it contains a transformer (e.g. ``DataFrameTransformer``) that converts the parts of the dataframe
    This is done here to reduce memory consumption: e.g., in case of OneHotEncoding, transforming creates lots of additional columns, and it's better to do it
    per batch, not per the whole dataset.

    If there is no transformer per data_frame, the corresponding data will not be included in batches
    """

    def __init__(self,
                 extractors: List[Extractor],
                 batching_strategy: Optional[BatcherStrategy] = None,
                 mini_batching_strategy: Optional[BatcherStrategy] = None,
                 ):
        if isinstance(extractors, int):
            raise ValueError('You try to pass a batch size to the Batcher. In the new version, the proper place for the batch size is task.settings.batch_size')
        self.extractors = extractors
        self.default_batching_strategy = SimpleBatcherStrategy()
        self.batching_strategy = batching_strategy or SimpleBatcherStrategy()
        self.mini_batching_strategy = mini_batching_strategy or SimpleBatcherStrategy()

    def preprocess_bundle(self, ibundle: IndexedDataBundle):
        for extractor in self.extractors:
            extractor.preprocess_bundle(ibundle)

    def fit_extract(self, batch_size: int, ibundle: IndexedDataBundle) -> IndexedDataBundle:
        index_df = self.get_batch_index(batch_size, ibundle, 0, False)
        ibundle = ibundle.change_index(index_df)
        for extractor in self.extractors:
            extractor.fit(ibundle)
        batch = Extractor.make_extraction(ibundle, self.extractors)
        return batch

    def _get_strategy(self, force_default_strategy):
        if force_default_strategy:
            return self.default_batching_strategy
        else:
            return self.batching_strategy

    def get_batch_count(self, batch_size:int,  db: IndexedDataBundle, force_default_strategy=False) -> int:
        return self._get_strategy(force_default_strategy).get_batch_count(batch_size, db.index_frame)

    def get_batch_index(self, batch_size: int, db: IndexedDataBundle, batch_index: int, force_default_strategy: bool) -> pd.DataFrame:
        index = self._get_strategy(force_default_strategy).get_batch(batch_size, db.index_frame, batch_index)
        index_df = db.index_frame.loc[index]
        return index_df

    def get_batch(self, batch_size: int, db: IndexedDataBundle, batch_index: int, force_default_strategy=False) -> IndexedDataBundle:
        """
        Args:
            db:
            batch_index:
            force_default_strategy:

        Returns: a dictionary with batch component, one per key of ``self.transformers``
        """
        index_df = self.get_batch_index(batch_size, db, batch_index, force_default_strategy)
        batch = Extractor.make_extraction(db.change_index(index_df), self.extractors)
        return batch

    def get_mini_batch_indices(self, mini_batch_size, batch: DataBundle) -> List[pd.Index]:
        n = self.mini_batching_strategy.get_batch_count(mini_batch_size, batch['index'])
        mini_batches = [self.mini_batching_strategy.get_batch(mini_batch_size, batch['index'], i) for i in range(n)]
        return mini_batches

    def get_mini_batch(self, index: pd.Index, batch: IndexedDataBundle) -> IndexedDataBundle:
        mini_batch = DataBundle()
        for key, df in batch.bundle.data_frames.items():
            if isinstance(df, pd.DataFrame):
                mini_batch[key] = df.loc[index]
            elif hasattr(df, 'sample_index'):
                mini_batch[key] = df.sample_index(index)
            else:
                raise ValueError(f"Unknown batch element type: {type(df)}")
        return IndexedDataBundle(batch.index_frame.loc[index], mini_batch)

    @staticmethod
    def generate_sample(
            bundle: Union[DataBundle, IndexedDataBundle],
            extractors: List[Extractor],
            batch_size = 10
    ):
        if isinstance(bundle, DataBundle):
            bundle = IndexedDataBundle(bundle['index'], bundle)
        batcher = Batcher(extractors)
        return batcher.fit_extract(batch_size, bundle)
