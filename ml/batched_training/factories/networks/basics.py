from typing import *
import pandas as pd
from ....single_frame_training import ModelConstructor
import torch
import copy
from yo_fluq_ds import Obj


class AnnotatedTensor:
    def __init__(self,
                 tensor: torch.Tensor,
                 dim_names: Iterable[str],
                 dim_indices: Optional[List[List]]
                 ):
        self.tensor = tensor
        self.dim_names = list(dim_names)
        self.dim_indices = dim_indices
        if self.dim_indices is not None:
            self.dim_reverse_indices = [
                pd.Series(range(len(s)), index=s)
                for s in self.dim_indices
            ]
        self.shape = tuple(tensor.shape)

    def sample_index(self, index: pd.Index):
        axis = self.dim_names.index(index.name)
        positions = self.dim_reverse_indices[axis].loc[index].values
        idx = [slice(None) if i != axis else positions for i in range(len(self.dim_names))]
        return AnnotatedTensor(self.tensor[idx], self.dim_names, None)


class CtorAdapter:
    def __init__(self,
                 type,
                 args_names = (),
                 **kwargs
                 ):
        self.type = type
        self.args_names = args_names
        self.kwargs = Obj(**kwargs)

    def __call__(self, *args, **kwargs):
        if len(args)!=len(self.args_names):
            raise ValueError(f'Expected {len(self.args_names)} argument: {self.args_names}, but received {len(args)}\n{args}')
        final_dictionary = {}
        for arg_name, arg in zip(self.args_names, args):
            final_dictionary[arg_name] = arg
        for key, value in self.kwargs.items():
            final_dictionary[key] = value
        for key, value in kwargs.items():
            final_dictionary[key] = value

        if isinstance(self.type, str):
            type = ModelConstructor._load_class(self.type)
        else:
            type = self.type

        return type(**final_dictionary)


def call_factory(factory, input):
    if isinstance(factory, torch.nn.Module):
        return copy.deepcopy(factory)
    elif callable(factory):
        return factory(input)
    else:
        raise ValueError(f'Factories are supposed to be callable (receiving a single argument) or torch.nn.Module, but was {type(factory)}\n{factory}')
