from typing import *

import pandas as pd
from .. import torch as btt
import torch


class EmbeddingHeadNetwork(torch.nn.Module):
    def __init__(self,
                 frame_name: str,
                 custom_embedding: Optional[torch.nn.Embedding] = None,
                 vocab_size: Optional[int] = None,
                 embedding_size: Optional[int] = None,
                ):
        super().__init__()
        self.frame_name = frame_name
        if custom_embedding is not None:
            self.embedding = custom_embedding
        elif vocab_size is not None and embedding_size is not None:
            self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        else:
            raise ValueError("Either custom embedding, or both vocab_size and embedding_size must be set")

    def forward(self, batch):
        x = batch[self.frame_name]
        if isinstance(x, btt.AnnotatedTensor):
            x = x.tensor
        elif isinstance(x, pd.DataFrame):
            x = btt.DfConversion.int(x)
        else:
            raise ValueError(f"Expected AnnotatedTensor or DataFrame, but was: {x}")
        x = self.embedding(x)
        x = x.squeeze()
        return x