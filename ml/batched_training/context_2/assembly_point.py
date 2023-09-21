from typing import *


import torch.nn

from ..factories import AssemblyPoint, DfConversion
from .components import Context2ExtractorToAggregatorFactory, Concatenate3DNetwork

from ..context import (
    ContextBuilder,
    ContextExtractor,
    Dim3NetworkFactory,
)

from .lstm_aggregator import LSTMAggregator
from .alignment_finalizer import AlignmentAggregationFinalizer



class ContextAssemblyPoint2(AssemblyPoint):
    def __init__(self,
                 units: Iterable[AssemblyPoint],
                 context_builder: ContextBuilder,
                 context_size: int = 3,
                 hidden_size: int = 20,
                 reverse_context_order: bool = False,
                 feature_transformer: Optional = None,
                 ):
        self.context_builder = context_builder
        self.context_size = context_size
        self.units = tuple(units)
        self.allowed_units = tuple(units)
        self.reverse_context_order = reverse_context_order
        self.feature_transformer = feature_transformer
        self.network_factory = Dim3NetworkFactory(None, self._get_first_layer)
        self.hidden_size = hidden_size

    def enable_units(self, units_to_enable: List[Union[str, AssemblyPoint]]):
        allowed_units = []
        for arg in units_to_enable:
            if isinstance(arg, AssemblyPoint):
                arg = arg.get_name()
            found = False
            for unit in self.units:
                print(arg, unit.get_name())
                if unit.get_name()==arg and unit not in allowed_units:
                    allowed_units.append(unit)
                    found = True
                    break
            if not found:
                raise ValueError(f'Value {arg} was not found among the units {[unit.get_name() for unit in self.units]}')
        self.allowed_units = tuple(allowed_units)

    def get_name(self):
        return ','.join(unit.get_name() for unit in self.allowed_units)

    def create_extractor(self):
        extractor_to_aggregator_factory = Context2ExtractorToAggregatorFactory(
            self.allowed_units,
            self.reverse_context_order,
            self.feature_transformer
        )

        finalizer = AlignmentAggregationFinalizer()

        context_extractor = ContextExtractor(
            self.get_name(),
            self.context_size,
            self.context_builder,
            extractor_to_aggregator_factory,
            finalizer
        )
        return context_extractor

    def _get_first_layer(self):
        if len(self.allowed_units) == 1:
            return self.allowed_units[0].create_network_factory()
        else:
            first_layer = Concatenate3DNetwork.Factory([u.create_network_factory() for u in self.allowed_units])
            return first_layer

    def create_network_factory(self):
        factory = self.network_factory.create_network_factory(self.hidden_size)
        return factory







