from typing import *
import pandas as pd
import torch
from .network_commons import TorchNetworkFactory, AnnotatedTensor
from .extracting_network import *

def _update_sizes_with_argument(argument_name, argument, sizes, modificator):
    if argument is None:
        return sizes
    elif isinstance(argument, torch.Tensor):
        return modificator(sizes, argument.shape[1])
    elif isinstance(argument, int):
        return modificator(sizes, argument)
    else:
        raise ValueError(f"Argument {argument_name} is supposed to be int, Tensor or none, but was {argument}")


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self,
                 sizes: List[int],
                 input: Union[None,torch.Tensor,int] = None,
                 output: Union[None, torch.Tensor, int] = None):
        super(FullyConnectedNetwork, self).__init__()
        sizes = _update_sizes_with_argument('input', input, sizes, lambda s, v: [v]+s)
        sizes = _update_sizes_with_argument('output', output, sizes, lambda s, v: s+[v])
        self.layers = torch.nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, input):
        X = input
        for layer in self.layers:
            X = layer(X)
            X = torch.sigmoid(X)
        return X

    @staticmethod
    def Factory(sizes: List[int], output: Union[None, torch.Tensor, int] = None):
        return UniversalFactory(
            FullyConnectedNetwork,
            'input',
            'N'+'-'.join([str(c) for c in sizes]),
            sizes = sizes, output=output
        )



class ParallelNetwork(torch.nn.Module):
    def __init__(self, **networks: torch.nn.Module):
        super(ParallelNetwork, self).__init__()
        self.networks = torch.nn.ModuleDict(networks)

    def forward(self, input):
        result = {}
        for name, network in self.networks.items():
            result[name] = network(input)
        return result

    class Factory(TorchNetworkFactory):
        def __init__(self, **factories: TorchNetworkFactory):
            self.factories = factories

        def create_network(self, task, input):
            networks = {}
            for key, factory in self.factories.items():
                networks[key] = factory.create_network(task, input)
            return ParallelNetwork(**networks)


