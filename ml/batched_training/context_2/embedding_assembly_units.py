from typing import *
from ..extractors import Extractor, IndexedDataBundle
from .embedding_extractors import NewEmbeddingExtractor, ExistingEmbeddingExtractor
from ..factories import Factories, Perceptron, InputConversionNetwork, DfConversion, AssemblyPoint
import pandas as pd
import numpy as np
import torch
from copy import deepcopy


class FourthDimSqueeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if len(input.shape)==4 and input.shape[2]==1:
            return input.squeeze()
        return input



class NewEmbeddingAssemblyUnit(AssemblyPoint):
    def __init__(self,
                 word_extractor: Extractor,
                 vocab_size: int,
                 dimensions: int,
                 unk_name = '<unk>'
                 ):
        self.word_extractor = word_extractor
        self.vocab_size = vocab_size
        self.dimensions = dimensions
        self.unk_name = unk_name

    def create_extractor(self):
        return NewEmbeddingExtractor(deepcopy(self.word_extractor), self.vocab_size, self.unk_name)

    def create_network_factory(self):
        factories = []
        factories.append(InputConversionNetwork(self.word_extractor.get_name(), conversion = DfConversion.int))
        factories.append(torch.nn.Embedding(self.vocab_size, self.dimensions))
        factories.append(FourthDimSqueeze())
        return Factories.FeedForward(*factories)

    def get_name(self):
        return self.word_extractor.get_name()

    def get_df_conversion(self):
        return DfConversion.int


class ExistingEmbeddingAssemblyUnit(AssemblyPoint):
    def __init__(self,
                 word_extractor: Extractor,
                 vectors: pd.DataFrame,
                 unk_name: str,
                 freeze: bool = False,
                 reduce_dimensionality_to: Optional[int] = None
                 ):
        self.word_extractor = word_extractor
        self.vectors = vectors
        self.unk_name = unk_name
        self.freeze = freeze
        self.reduce_dimensionality_to = reduce_dimensionality_to

    def create_extractor(self):
        return ExistingEmbeddingExtractor(
            deepcopy(self.word_extractor),
            self.vectors,
            self.unk_name
        )

    def create_network_factory(self):
        tensor = torch.tensor(self.vectors.values.astype(float)).float()
        factories = []
        factories.append(InputConversionNetwork(self.word_extractor.get_name(), conversion=DfConversion.int))
        emb = torch.nn.Embedding.from_pretrained(tensor)
        if self.freeze:
            emb.requires_grad_(False)
        factories.append(Factories.Factory(emb))
        factories.append(FourthDimSqueeze())
        if self.reduce_dimensionality_to is not None:
            factories.append(Factories.Factory(Perceptron, output_size=self.reduce_dimensionality_to))
        return Factories.FeedForward(*factories)

    def get_name(self):
        return self.word_extractor.get_name()

    def get_df_conversion(self):
        return DfConversion.int

