from typing import *

import torch
import pandas as pd

from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss

from ..model_handler import BatchedModelHandler
from ...training_core import AbstractTrainingTask



class MultiLayerPerceptronWithParsing(torch.nn.Module):
    def __init__(self, sample_features: Dict[str, pd.DataFrame], accepted_inputs: List[str], sizes: List[int]):
        super(MultiLayerPerceptronWithParsing, self).__init__()
        self.accepted_inputs = accepted_inputs
        sample_tensor = self.get_tensor(sample_features)
        self.layers = torch.nn.ModuleList()
        sizes = [sample_tensor.shape[1]] + sizes
        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))

    def get_tensor(self, features: Dict[str, pd.DataFrame]):
        tensors = []
        for input_name in self.accepted_inputs:
            tensor = torch.tensor(features[input_name].astype(float).values).float()
            tensors.append(tensor)
        tensor = torch.cat(tensors, 1)
        return tensor

    def forward(self, features: Dict[str, pd.DataFrame]):
        X = self.get_tensor(features)
        for layer in self.layers:
            # print(X.shape, layer)
            X = layer(X)
            X = torch.sigmoid(X)
        return X


class TorchModel:
    def __init__(self, network: torch.nn.Module, optimizer: Optional[Optimizer] = None, loss: Optional[_Loss] = None):
        self.network = network
        self.optimizer = optimizer or torch.optim.SGD(network.parameters(), lr=0.1)
        self.loss = loss or torch.nn.MSELoss()


class TorchClassificationModelHandler(BatchedModelHandler):
    def __init__(self, labels: str, initializer: Callable[[AbstractTrainingTask, Dict[str, pd.DataFrame]], TorchModel]):
        self.labels = labels
        self.initializer = initializer
        self.model = None  # type: Optional[TorchModel]

    def instantiate(self, task, input):
        self.model = self.initializer(task, input)

    def _get_xy(self, input):
        return {key: value for key, value in input.items() if key != self.labels}, input[self.labels]

    def train(self, input: Dict[str, pd.DataFrame]):
        X, y = self._get_xy(input)
        self.model.optimizer.zero_grad()
        output = self.model.network(X)
        targets = torch.tensor(y.values).float()
        loss = self.model.loss(output, targets)
        loss.backward()
        self.model.optimizer.step()
        return loss.item()

    def predict(self, input: Dict[str, pd.DataFrame]):
        X, y = self._get_xy(input)
        output = self.model.network(X)
        output = output.flatten().tolist()
        df = pd.DataFrame(dict(predicted=output, true=y[y.columns[0]]))
        return df
