from typing import *
import torch
import pandas as pd


class Perceptron(torch.nn.Module):
    def __init__(self, input_size: Union[torch.Tensor, int], output_size: int):
        super(Perceptron, self).__init__()
        i_size = Perceptron.tensor_argument_to_int(input_size)
        o_size = Perceptron.tensor_argument_to_int(output_size)
        self.linear_layer = torch.nn.Linear(i_size, o_size)

    def forward(self, input):
        return torch.sigmoid(self.linear_layer(input))

    @staticmethod
    def tensor_argument_to_int(argument):
        if isinstance(argument, torch.Tensor):
            return argument.shape[1]
        elif isinstance(argument, pd.DataFrame):
            return argument.shape[1]
        elif isinstance(argument, int):
            return argument
        else:
            raise ValueError(f'Expected tensor, dataframe or int, but was: {argument}')

