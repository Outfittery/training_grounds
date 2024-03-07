from typing import *
from ... import batched_training as bt
from .. import context as btc
from .. import torch as btt
import torch
from .feature_head_network import FeatureHeadNetwork
from .embedding_head_network import EmbeddingHeadNetwork
from .context_head_network import ContextHeadNetwork
from . import dim_networks as dn
from enum import Enum
from dataclasses import dataclass, field
from .tensorlike_extractor_to_aggregator import TensorlikeExtractorToAggregator


@dataclass
class Gorynych:

    class HeadType(Enum):
        Features = 0
        Embedding = 1
        Context = 2

    @dataclass
    class RegisterItem:
        head_type: 'Gorynych.HeadType'
        vocab_size: Optional[int] = None
        ignore: bool = False


    head_output_size: int = 50
    new_embedding_vector_size = 30
    extractors_register: Dict[str,'Gorynych.RegisterItem'] = field(default_factory=dict)
    context_dimensionality_reduction_type: dn.Dim3NetworkType = dn.Dim3NetworkType.AlonAttention
    context_size: int = 5

    def create_context_extractor_from_inner_extractors(self,
                                                       name: str,
                                                       extractors: Iterable[bt.Extractor],
                                                       context_builder: btc.ContextBuilder
                                                       ):
        conversions = {}
        for ex in extractors:
            if self.get_register_item(ex).head_type == Gorynych.HeadType.Embedding:
                conversions[ex.get_name()] = btt.DfConversion.int
        extractor_to_aggregator_factory = TensorlikeExtractorToAggregator(
            extractors,
            conversions,
            False,
            None
        )
        context_extractor = btc.ContextExtractor(
            name,
            self.context_size,
            context_builder,
            extractor_to_aggregator_factory,
            btc.AlignmentAggregationFinalizer(use_dict_if_one_tensor=True)
        )
        return context_extractor


    def register(self,
                 extractor: bt.Extractor,
                 type: 'Gorynych.HeadType',
                 vocab_size: Optional[int] = None):
        self.extractors_register[extractor.get_name()]=Gorynych.RegisterItem(type, vocab_size)


    def get_register_item(self, extractor: bt.Extractor) -> 'Gorynych.RegisterItem':
        if extractor.get_name() in self.extractors_register:
            return self.extractors_register[extractor.get_name()]
        type = Gorynych.HeadType.Features
        if isinstance(extractor, btc.ContextExtractor):
            type = Gorynych.HeadType.Context
        if isinstance(extractor, btc.AbstractEmbeddingExtractor):
            type = Gorynych.HeadType.Embedding
        return Gorynych.RegisterItem(type)


    def create_embedding_head(self,
                              sample: bt.DataBundle,
                              extractor: bt.Extractor
                              ):
        if isinstance(extractor, btc.ExistingEmbeddingExtractor):
            emb = torch.nn.Embedding.from_pretrained(btt.DfConversion.float(extractor.vectors))
            return EmbeddingHeadNetwork(extractor.get_name(), emb)
        if isinstance(extractor, btc.NewEmbeddingExtractor):
            return EmbeddingHeadNetwork(extractor.get_name(), None, extractor.vocab_size, self.new_embedding_vector_size)
        reg_item = self.get_register_item(extractor)
        if reg_item.vocab_size is not None:
            return EmbeddingHeadNetwork(extractor.get_name(), None, reg_item.vocab_size, self.new_embedding_vector_size)
        raise ValueError(f'Cannot create head for extractor {extractor}: embedding is requested, but unable to locate vocabulary size')

    def create_feature_head(self,
                            sample: bt.DataBundle,
                            extractor: bt.Extractor,
                            skip_transformation: bool = False,
                            output_size: Optional[int] = None
                            ):
        if skip_transformation:
            return FeatureHeadNetwork(sample, extractor.get_name(), out_feature_count = None, non_linear = None)
        else:
            if output_size is None:
                output_size = self.head_output_size
            return FeatureHeadNetwork(sample, extractor.get_name(), out_feature_count = output_size)

    def create_context_head(self,
                            sample: bt.DataBundle,
                            extractor: btc.ContextExtractor
                            ):
        heads = []
        for ea in extractor.extractors_and_aggregators:
            inner_extractor = ea.extractor
            reg_item = self.get_register_item(inner_extractor)
            if reg_item.ignore:
                continue
            if reg_item.head_type == Gorynych.HeadType.Embedding:
                heads.append(self.create_embedding_head(sample, inner_extractor))
            elif reg_item.head_type == Gorynych.HeadType.Features:
                heads.append(self.create_feature_head(sample, inner_extractor, True))
            else:
                raise ValueError(f'Inner extractor {inner_extractor} in context extractor {extractor} requested a head type {type}, which is not possible')

        head_tensors = [head(sample) for head in heads]
        input = torch.cat(head_tensors, dim=2)
        network_type = self.context_dimensionality_reduction_type
        if network_type == dn.Dim3NetworkType.LSTM:
            reductor = dn.LSTMNetwork(input, self.head_output_size)
        elif network_type == dn.Dim3NetworkType.AlonAttention:
            reductor = dn.AlonAttention(input, hidden_size=self.head_output_size)
        elif network_type == dn.Dim3NetworkType.AlonAttentionSigmoid:
            reductor = dn.AlonAttention(input, hidden_size=self.head_output_size, sigmoid=True)
        elif network_type == dn.Dim3NetworkType.AlonAttentionWithoutFullyConnected:
            reductor = dn.AlonAttention(input, hidden_size=None)
        elif network_type == dn.Dim3NetworkType.AlonAttentionWithoutFullyConnectedSigmoid:
            reductor = dn.AlonAttention(input, hidden_size=None, sigmoid=True)
        elif network_type == dn.Dim3NetworkType.SelfAttentionAndLSTM:
            reductor = dn.AttentionReccurentNetwork(input, hidden_size=self.head_output_size)
        else:
            raise ValueError(f"Unknown network type {network_type}")
        return ContextHeadNetwork(heads, reductor)


    def create_head(self, sample: bt.DataBundle, extractor: bt.Extractor):
        reg_item = self.get_register_item(extractor)
        if reg_item.ignore:
            return None
        if reg_item.head_type == Gorynych.HeadType.Embedding:
            return self.create_embedding_head(sample, extractor)
        elif reg_item.head_type == Gorynych.HeadType.Features:
            return self.create_feature_head(sample, extractor)
        elif reg_item.head_type == Gorynych.HeadType.Context:
            return self.create_context_head(sample, extractor)
        else:
            raise ValueError(f"Unknown type {reg_item.head_type} for extractor {extractor.get_name()}")






