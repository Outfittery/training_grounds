from typing import *

import pandas as pd
import torch

from .network_commons import TorchNetworkFactory, AnnotatedTensor


def df_to_torch(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.astype(float).values).float()


class PrependableFactory:
    def prepend_extraction(self,
                           input_frames: Union[None, str, List[str]],
                           raise_if_inputs_are_missing=False) -> TorchNetworkFactory:
        return FeedForwardNetwork.Factory(
            ExtractingNetwork.Factory(input_frames, raise_if_inputs_are_missing),
            self
        )


class UniversalFactory(TorchNetworkFactory, PrependableFactory):
    def __init__(self, type, input_arg_mapping, **kwargs):
        self.type = type
        self.input_arg_mapping = input_arg_mapping
        self.kwargs = kwargs

    def create_network(self, task, input):
        kwargs = {k: v for k, v in self.kwargs.items()}
        if self.input_arg_mapping is not None:
            kwargs[self.input_arg_mapping] = input
        return self.type(**kwargs)





class ExtractingNetwork(torch.nn.Module):
    def __init__(self,
                 input_frames: Union[None, str, List[str]],
                 raise_if_inputs_are_missing=False
                 ):
        super(ExtractingNetwork, self).__init__()
        if isinstance(input_frames, list):
            pass
        elif isinstance(input_frames, str):
            input_frames = [input_frames]
        else:
            raise ValueError(f'Expected str or List[str], got {type(input_frames)}, value {input_frames}')
        self.input_frames = input_frames
        self.raise_if_inputs_are_missing = raise_if_inputs_are_missing

    @staticmethod
    def collect_tensors(input, input_frames, raise_if_inputs_are_missing):
        en = input_frames
        tensors = []
        for frame in en:
            if frame not in input:
                if raise_if_inputs_are_missing:
                    raise ValueError(f'Missing frame {frame} in batch')
                else:
                    continue
            if isinstance(input[frame], pd.DataFrame):
                tensors.append(df_to_torch(input[frame]))
            elif isinstance(input[frame], AnnotatedTensor):
                tensors.append(input[frame].tensor)
            elif isinstance(input[frame], torch.Tensor):
                tensors.append(input[frame])
            else:
                raise ValueError(
                    f'Batch element must be torch.Tensor, pandas.Dataframe or AnnotatedTensor, but was {type(input[frame])}')

        return tensors

    def forward(self, input: Dict[str, pd.DataFrame]):
        if isinstance(input, torch.Tensor):
            if self.input_frames is not None:
                raise ValueError('The input was tensor, but `input_frames` were provided, suggesting it would be a dictionary')

        if self.input_frames is None:
            en = list(input.keys())
        else:
            en = self.input_frames
        tensors = ExtractingNetwork.collect_tensors(input, en, self.raise_if_inputs_are_missing)

        if len(tensors) == 0:
            raise ValueError(f'No tensors were produced. Input keys are {list(input.keys())}, expected keys are {list(en)}')
        if len(tensors) == 1:
            return tensors[0]
        else:
            return torch.cat(tensors, 1)

    @staticmethod
    def Factory(input_frames: Union[None, str, List[str]], raise_if_inputs_are_missing=False):
        return UniversalFactory(
            ExtractingNetwork,
            None,
            input_frames=input_frames,
            raise_if_inputs_are_missing=raise_if_inputs_are_missing
        )


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, *networks):
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

    class Factory(TorchNetworkFactory):
        def __init__(self, *factories: TorchNetworkFactory):
            self.factories = factories

        def create_network(self, task, input):
            networks = []
            for factory in self.factories:
                network = factory.create_network(task, input)
                if network is not None:
                    networks.append(network)
                    input = network(input)

            return FeedForwardNetwork(*networks)

        def preview_batch(self, input):
            for factory in self.factories:
                factory.preview_batch(input)
