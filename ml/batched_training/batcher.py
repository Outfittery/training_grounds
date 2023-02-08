from typing import *

import pandas as pd

from .samplers import Sampler, SequencialSampler
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
                 extractors: List[Extractor],
                 training_sampler: Optional[Sampler] = None,
                 mini_batching_sampler: Optional[Sampler] = None,
                 ):
        if isinstance(extractors, int):
            raise ValueError('You try to pass a batch size to the Batcher. In the new version, the proper place for the batch size is task.settings.batch_size')
        self.extractors = extractors
        self.inference_sampler = SequencialSampler()
        self.training_sampler = training_sampler if training_sampler is not None else SequencialSampler()
        self.mini_batching_sampler = mini_batching_sampler if mini_batching_sampler is not None else SequencialSampler()

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

    def _get_strategy(self, in_inference):
        if in_inference:
            return self.inference_sampler
        else:
            return self.training_sampler

    def get_batch_count(self, batch_size:int,  db: IndexedDataBundle, in_inference=False) -> int:
        return self._get_strategy(in_inference).get_batch_count(batch_size, db)

    def get_batch_index(self, batch_size: int, db: IndexedDataBundle, batch_index: int, in_inference: bool) -> pd.DataFrame:
        index_df = self._get_strategy(in_inference).get_batch_index_frame(batch_size, db, batch_index)
        return index_df

    def get_batch(self, batch_size: int, db: IndexedDataBundle, batch_index: int, in_inference=False) -> IndexedDataBundle:
        """
        Args:
            db:
            batch_index:
            force_default_strategy:

        Returns: a dictionary with batch component, one per key of ``self.transformers``
        """
        index_df = self.get_batch_index(batch_size, db, batch_index, in_inference)
        batch = Extractor.make_extraction(db.change_index(index_df), self.extractors)
        return batch

    def get_mini_batch_indices(self, mini_batch_size, batch: DataBundle) -> List[pd.Index]:
        n = self.mini_batching_sampler.get_batch_count(mini_batch_size, batch)
        mini_batches = [self.mini_batching_sampler.get_batch_index_frame(mini_batch_size, batch, i).index for i in range(n)]
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
