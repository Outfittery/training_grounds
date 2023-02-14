from typing import *
import torch
from .basics import call_factory

class ParallelNetwork(torch.nn.Module):
    def __init__(self, **networks: torch.nn.Module):
        super(ParallelNetwork, self).__init__()
        self.networks = torch.nn.ModuleDict(networks)
        self.debug = False

    def forward(self, input):
        results = {}
        if self.debug:
            self.intermediate_values = results
        for key, network in self.networks.items():
            results[key] = network
        return results

    class Factory:
        def __init__(self, **factories: Union[torch.nn.Module, Callable]):
            self.factories = factories
            self.debug = False

        def __call__(self, input):
            networks = {}
            for key, factory in self.factories.items():
                try:
                    network = call_factory(factory, input)
                    networks[key] = network
                except Exception as ex:
                    raise ValueError(f'Error when initializing FeedForwardFactory at step {key}') from ex
            network = ParallelNetwork(**networks)
            network.debug = self.debug
            return network
