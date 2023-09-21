from typing import *
import torch
from .basics import call_factory


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, *networks: torch.nn.Module):
        super(FeedForwardNetwork, self).__init__()
        self.networks = torch.nn.ModuleList(networks)
        self.debug = False

    def forward(self, input):
        results = []
        if self.debug:
            self.intermediate_values = results
        results.append(input)
        for network in self.networks:
            input = network(input)
            results.append(input)
        return input

    class Factory:
        def __init__(self, *factories: Union[torch.nn.Module, Callable]):
            self.factories = factories
            self.debug = False

        def __call__(self, input):
            networks = []
            for index, factory in enumerate(self.factories):
                try:
                    network = call_factory(factory, input)
                    networks.append(network)
                    input = network(input)
                except Exception as ex:
                    raise ValueError(f'Error when initializing FeedForwardFactory at step {index}, factory {factory}') from ex
            network = FeedForwardNetwork(*networks)
            network.debug = self.debug
            return network


