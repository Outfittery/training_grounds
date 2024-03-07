from typing import *

import pandas as pd

from .. import torch as btt
import torch


class FeatureHeadNetwork(torch.nn.Module):
    def __init__(self,
                 sample,
                 frame_name: str,
                 out_feature_count: Optional[int] = None,
                 conversion: Callable = btt.DfConversion.auto,
                 non_linear: Optional[Callable] = torch.sigmoid,
                ):
        super().__init__()
        self.frame_name = frame_name
        self.conversion = conversion
        if out_feature_count is not None:
            self.linear = torch.nn.Linear(sample[self.frame_name].shape[1], out_feature_count)
        else:
            self.linear = None
        self.non_linear = non_linear

    def forward(self, batch):
        x = batch[self.frame_name]
        if isinstance(x, btt.AnnotatedTensor):
            x = x.tensor
        elif isinstance(x, pd.DataFrame):
            x = self.conversion(x)
        else:
            raise ValueError(f"Input should be AnnotatedTensor or DataFrame, but was {x}")
        if self.linear is not None:
            x = self.linear(x)
        if self.non_linear is not None:
            x = self.non_linear(x)
        return x