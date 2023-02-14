from typing import *

import pandas as pd
import copy

from yo_fluq_ds import KeyValuePair, Obj

from ..extractors import Extractor
from ..data_bundle import IndexedDataBundle


class ContextBuilder:
    def fit(self, ibundle: IndexedDataBundle):
        pass

    def build_context(self, ibundle: IndexedDataBundle, context_size) -> pd.DataFrame:
        raise NotImplementedError()


class ContextAggregator:
    def fit(self, features_df: pd.DataFrame):
        pass

    def aggregate_context(self, features_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


class ExtractorInnerData:
    def __init__(self, ibundle: IndexedDataBundle, context_size: int):
        self.context_size = context_size
        self.ibundle = ibundle
        self.context_df = None  # type: Optional[pd.DataFrame]
        self.feature_dfs = {}  # type: Dict[str,pd.DataFrame]
        self.agg_dfs = {}  # type: Dict[str,pd.DataFrame]
        self.result_df = None  # type: Optional[pd.DataFrame]


class ExtractorToAggregator:
    def __init__(self, extractor: Extractor, aggregators: List[ContextAggregator]):
        self.extractor = extractor
        self.aggregators = aggregators

    @staticmethod
    def apply_several(context_ibundle: IndexedDataBundle, data, extractors_and_aggregators: List['ExtractorToAggregator']):
        for extractor_index, ea in enumerate(extractors_and_aggregators):
            fdf = ea.extractor.extract(context_ibundle)
            data.feature_dfs[f'f{extractor_index}'] = fdf
            for aggregator_index, a in enumerate(ea.aggregators):
                adf = a.aggregate_context(fdf)
                data.agg_dfs[f'f{extractor_index}a{aggregator_index}'] = adf


class ExtractorToAggregatorFactory:
    def create_extractors_and_aggregators(self, ibundle: IndexedDataBundle) -> List[ExtractorToAggregator]:
        raise NotImplementedError()


class SimpleExtractorToAggregatorFactory(ExtractorToAggregatorFactory):
    def __init__(self, extractor, *aggregators):
        self.extractor = extractor
        self.aggregators = [copy.deepcopy(c) for c in aggregators]

    def create_extractors_and_aggregators(self, ibundle: IndexedDataBundle) -> List[ExtractorToAggregator]:
        self.extractor.fit(ibundle)
        features_df = self.extractor.extract(ibundle)
        for a in self.aggregators:
            a.fit(features_df)
        return [ExtractorToAggregator(self.extractor, self.aggregators)]


class AggregationFinalizer:
    def fit(self, index: pd.DataFrame, features: Dict[str, pd.DataFrame], aggregations: Dict[str, pd.DataFrame]):
        pass

    def finalize(self, index: pd.DataFrame, features: Dict[str, pd.DataFrame], aggregations: Dict[str, pd.DataFrame]):
        raise NotImplementedError()


class ContextExtractor(Extractor):
    def __init__(self,
                 name: str,
                 context_size: int,
                 context_builder: ContextBuilder,
                 feature_extractor_factory: ExtractorToAggregatorFactory,
                 finalizer: AggregationFinalizer,
                 debug=False
                 ):
        self.name = name
        self.context_size = context_size
        self.context_builder = context_builder
        self.feature_extractor_factory = feature_extractor_factory
        self.feature_extractor = None  # type: Optional[Extractor]
        self.finalizer = finalizer
        self.debug = debug
        self.extractors_and_aggregators = None

    def get_name(self):
        return self.name

    def _extract_till_finalization(self, data: ExtractorInnerData):
        data.context_df = self.context_builder.build_context(data.ibundle, self.context_size)
        if data.context_df.shape[0] != 0:
            ExtractorToAggregator.apply_several(data.ibundle.change_index(data.context_df), data, self.extractors_and_aggregators)

    def fit(self, ibundle: IndexedDataBundle):
        fit_data = Obj()
        if self.debug:
            self.fit_data_ = fit_data
        self.context_builder.fit(ibundle)
        fit_data['context_index_frame'] = self.context_builder.build_context(ibundle, self.context_size)
        self.extractors_and_aggregators = self.feature_extractor_factory.create_extractors_and_aggregators(ibundle.change_index(fit_data['context_index_frame']))
        data = ExtractorInnerData(ibundle, self.context_size)
        if self.debug:
            self.data_ = data
        self._extract_till_finalization(data)
        self.finalizer.fit(ibundle.index_frame, data.feature_dfs, data.agg_dfs)

    def extract(self, ibundle: IndexedDataBundle) -> pd.DataFrame:
        if self.extractors_and_aggregators is None:
            raise ValueError('extractors_and_aggregators is None: have you forgotten to fit?')
        if ibundle.index_frame.index.duplicated().any():
            raise ValueError('This extractor cannot function with duplicating rows in the sample. Check your batcher, does it produce deduplicated indices?')
        data = ExtractorInnerData(ibundle, self.context_size)
        if self.debug:
            self.data_ = data
        self._extract_till_finalization(data)
        data.result_df = self.finalizer.finalize(ibundle.index_frame, data.feature_dfs, data.agg_dfs)
        return data.result_df
