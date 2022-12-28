from .. import factories as btf
from .lstm_components import  LSTMNetwork
import torch
from functools import partial

class Dim3NetworkFactory:
    def __init__(self, input_name: str):
        self.input_name = input_name
        self.droupout_rate = None

    def create_network(self, input, hidden_size):
        factories = []
        factories.append(btf.InputConversionNetwork(self.input_name))
        if self.droupout_rate is not None:
            factories.append(torch.nn.Dropout3d(self.droupout_rate))
        factories.append(partial(LSTMNetwork, hidden_size = hidden_size))
        pipeline = btf.FeedForwardNetwork.Factory(*factories)
        return pipeline(input)


class PivotNetworkFactory:
    def __init__(self, input_name: str):
        self.input_name = input_name
        self.network_sizes = (20,)

    def create_network(self, input, hidden_size):
        return btf.Factories.FullyConnected(hidden_size, self.input_name)(input)