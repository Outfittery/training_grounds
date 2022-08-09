from typing import *

from .. import torch as btt
from ... import batched_training as bt


class ExtractorNetworkBinding:
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.from_basis_task = None

    def get_name(self):
        return self.name

    def create_network_factory(self, task, batch):
        if not self.enabled:
            return None
        # TODO: process basis tasks here
        return self._create_network_factory_internal(task, batch)

    def create_extractor(self, task, bundle):
        if not self.enabled:
            return None
        # TODO: process basis tasks here
        return self._create_extractor_internal(task, bundle)

    def _create_network_factory_internal(self, task, batch):
        raise NotImplementedError()

    def _create_extractor_internal(self, task, bundle):
        raise NotImplementedError()


class MirrorSettings:
    def __init__(self,
                 tail_network_factories: List[btt.TorchNetworkFactory],
                 label_extractor,
                 bindings: List[ExtractorNetworkBinding],
                 ):
        self.tail_network_factories = tail_network_factories
        self.label_extractor = label_extractor
        self.bindings = bindings


class _MirrorExtractorFactory(btt.TorchExtractorFactory):
    def __init__(self, settings: MirrorSettings):
        self.settings = settings

    def create_extractors(self, task, bundle) -> List[bt.Extractor]:
        extractors = [c.create_extractor(task, bundle) for c in self.settings.bindings]
        extractors.append(self.settings.label_extractor)
        return extractors


class _MirrorNetworkFactory(btt.TorchNetworkFactory):
    def __init__(self, settings: MirrorSettings):
        self.settings = settings

    def create_network(self, task, input):
        first_layer_networks = {c.get_name(): c.create_network_factory(task, input) for c in self.settings.bindings}
        factory = btt.FeedForwardNetwork.Factory(
            btt.ParallelNetwork.Factory(**first_layer_networks),
            *self.settings.tail_network_factories
        )
        factory.preview_batch(input)
        network = factory.create_network(task, input)
        return network
