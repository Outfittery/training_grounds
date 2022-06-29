from typing import *

import torch
import pandas as pd


class TorchNetworkFactory:
    def create_network(self, task, input):
        raise NotImplementedError()


class AnnotatedTensor:
    def __init__(self,
                 tensor: torch.Tensor,
                 dim_names: List[str],
                 dim_indices: Optional[List[List]]
                 ):
        self.tensor = tensor
        self.dim_names = dim_names
        self.dim_indices = dim_indices
        if self.dim_indices is not None:
            self.dim_reverse_indices = [
                pd.Series(range(len(s)), index=s)
                for s in self.dim_indices
            ]

    def sample_index(self, index: pd.Index):
        axis = self.dim_names.index(index.name)
        positions = self.dim_reverse_indices[axis].loc[index].values
        idx = [slice(None) if i != axis else positions for i in range(len(self.dim_names))]
        return AnnotatedTensor(self.tensor[idx], self.dim_names, None)
