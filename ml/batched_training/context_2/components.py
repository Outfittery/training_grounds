from typing import *


import torch.nn

from ..factories import AssemblyPoint, DfConversion
from ..extractors import IndexedDataBundle
from ..context import ExtractorToAggregatorFactory, ExtractorToAggregator

from .lstm_aggregator import LSTMAggregator


class Context2ExtractorToAggregatorFactory(ExtractorToAggregatorFactory):
    def __init__(self,
                 assembly_points: Iterable[AssemblyPoint],
                 reverse_context_order: bool = False,
                 feature_transformer: Optional = None
                 ):
        self.assembly_point = tuple(assembly_points)
        self.reverse_context_order = reverse_context_order
        self.feature_transformer = feature_transformer

    def create_extractors_and_aggregators(self, ibundle: IndexedDataBundle) -> List[ExtractorToAggregator]:
        result = []
        for ap in self.assembly_point:
            ex = ap.create_extractor()
            ex.fit(ibundle)
            input = ex.extract(ibundle)
            conversion = DfConversion.auto
            if hasattr(ap, 'get_df_conversion'):
                conversion = ap.get_df_conversion()
            aggregator = LSTMAggregator(
                conversion = conversion,
                feature_transformer = self.feature_transformer,
                reverse_context_order = self.reverse_context_order
            )
            aggregator.fit(ibundle.index_frame, input)
            result.append(ExtractorToAggregator(ex, [aggregator]))
        return result


class Concatenate3DNetwork(torch.nn.Module):
    def __init__(self, inner_networks):
        super().__init__()
        self.inner_networks = torch.nn.ModuleList(inner_networks)

    def forward(self, input):
        tensors = [network(input) for network in self.inner_networks]
        result = torch.cat(tensors, dim=2)
        return result

    class Factory:
        def __init__(self, factories):
            self.factories = factories

        def __call__(self, db):
            return Concatenate3DNetwork([f(db) for f in self.factories])