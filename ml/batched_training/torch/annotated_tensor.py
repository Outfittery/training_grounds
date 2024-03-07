from typing import *
import torch
import pandas as pd


class AnnotatedTensor:
    def __init__(self,
                 tensor: torch.Tensor,
                 dim_names: Iterable[str],
                 dim_indices: Optional[List[List]]
                 ):
        self.tensor = tensor
        self.dim_names = tuple(dim_names)
        self.dim_indices = tuple(tuple(z) for z in dim_indices) if dim_indices is not None else None
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

