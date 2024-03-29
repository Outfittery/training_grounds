from .. import factories as btf
from .lstm_components import  LSTMNetwork
from .alon_attention import AlonAttention
from .self_attention import AttentionReccurentNetwork
import torch
from functools import partial
from enum import Enum

class Dim3NetworkType(Enum):
    LSTM = 0
    AlonAttention = 1
    AlonAttentionSigmoid = 2
    AlonAttentionWithoutFullyConnected = 3
    AlonAttentionWithoutFullyConnectedSigmoid = 4
    SelfAttentionAndLSTM = 5

class Dim3NetworkFactory:
    def __init__(self,
                 input_name: str,
                 get_custom_input_layer = None
                 ):
        self.input_name = input_name
        self.droupout_rate = None
        self.network_type = Dim3NetworkType.LSTM
        self.get_custom_input_layer = get_custom_input_layer

    def create_network_factory(self, hidden_size):
        factories = []
        if self.get_custom_input_layer is None:
            factories.append(btf.InputConversionNetwork(self.input_name))
        else:
            factories.append(self.get_custom_input_layer())

        if self.droupout_rate is not None:
            factories.append(torch.nn.Dropout3d(self.droupout_rate))

        if self.network_type == Dim3NetworkType.LSTM:
            factories.append(partial(LSTMNetwork, hidden_size=hidden_size))
        elif self.network_type == Dim3NetworkType.AlonAttention:
            factories.append(partial(AlonAttention, hidden_size=hidden_size))
        elif self.network_type == Dim3NetworkType.AlonAttentionSigmoid:
            factories.append(partial(AlonAttention, hidden_size=hidden_size, sigmoid=True))
        elif self.network_type == Dim3NetworkType.AlonAttentionWithoutFullyConnected:
            factories.append(partial(AlonAttention, hidden_size=None))
        elif self.network_type == Dim3NetworkType.AlonAttentionWithoutFullyConnectedSigmoid:
            factories.append(partial(AlonAttention, hidden_size=None, sigmoid=True))
        elif self.network_type == Dim3NetworkType.SelfAttentionAndLSTM:
            factories.append(partial(AttentionReccurentNetwork, hidden_size=hidden_size))

        pipeline = btf.FeedForwardNetwork.Factory(*factories)
        return pipeline

    def create_network(self, input, hidden_size):
        pipeline = self.create_network_factory(hidden_size)
        return pipeline(input)


class PivotNetworkFactory:
    def __init__(self, input_name: str):
        self.input_name = input_name
        self.network_sizes = (20,)

    def create_network(self, input, hidden_size):
        return btf.Factories.FullyConnected(hidden_size, self.input_name)(input)