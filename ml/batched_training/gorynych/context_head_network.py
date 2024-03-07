from typing import *
import torch

class ContextHeadNetwork(torch.nn.Module):
    def __init__(self,
                 pre_heads: Iterable[torch.nn.Module],
                 dim_reductor: torch.nn.Module,
                 ):
        super().__init__()
        self.pre_heads = torch.nn.ModuleList(pre_heads)
        self.dim_reductor = dim_reductor

    def forward(self, input):
        tensors = [pre_head(input) for pre_head in self.pre_heads]
        tensor = torch.cat(tensors, dim=2)
        return self.dim_reductor(tensor)