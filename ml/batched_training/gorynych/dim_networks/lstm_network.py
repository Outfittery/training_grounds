from typing import *
import torch

class LSTMNetwork(torch.nn.Module):
    def __init__(self, context_size: Union[int, torch.Tensor], hidden_size: Union[int, Tuple, List[int]]):
        super(LSTMNetwork, self).__init__()
        if isinstance(context_size, torch.Tensor):
            context_size = context_size.shape[2]
        if not isinstance(hidden_size, int):
            hidden_size = hidden_size[0]
        self.lstm = torch.nn.LSTM(
            context_size,
            hidden_size
        )

    def forward(self, input):
        lstm_output = self.lstm(input)
        output = lstm_output[1][0]
        output = output.reshape(output.shape[1], output.shape[2])
        return output
