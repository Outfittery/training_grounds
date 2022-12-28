from typing import *
from ... import batched_training as bt
import pandas as pd
from .conventions import Conventions
import torch
from yo_fluq_ds import Obj



class TorchModelHandler(bt.BatchedModelHandler):
    def __init__(self,
                 network_factory: Callable,
                 optimizer_factory: Callable,
                 loss_factory: Callable
                 ):
        self.network_factory = network_factory
        self.optimizer_factory = optimizer_factory
        self.loss_factory = loss_factory


    def instantiate(self, task, input: Dict[str, pd.DataFrame]) -> None:
        self.network = self.network_factory(input)
        self.optimizer = self.optimizer_factory(self.network.parameters())
        self.loss = self.loss_factory()

    def _predict_1_dim(self, input, labels):
        output = self.network(input)
        output = output.flatten().tolist()
        result = input['index'].copy()
        result['true'] = labels[labels.columns[0]]
        result['predicted'] = output
        return result

    def _predict_multi_dim(self, input, labels):
        result = input['index'].copy()
        output = self.network(input)
        for i, c in enumerate(labels.columns):
            result['true_' + c] = labels[c]
            result['predicted_' + c] = output[:, i].tolist()
        return result

    def predict(self, input: Dict[str, pd.DataFrame]):
        self.network.eval()
        labels = input[Conventions.LabelFrame]
        if labels.shape[1] == 1:
            return self._predict_1_dim(input, labels)
        else:
            return self._predict_multi_dim(input, labels)

    def _train_1_dim(self, input, labels):
        self.optimizer.zero_grad()
        result = self.network(input).flatten()
        target = torch.tensor(input[Conventions.LabelFrame].values).float().flatten()
        loss = self.loss(result, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_multi_dim(self, input, labels):
        self.optimizer.zero_grad()
        result = self.network(input)
        target = torch.tensor(input[Conventions.LabelFrame].values).float()
        loss = self.loss(result, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, input: Dict[str, pd.DataFrame]) -> float:
        self.network.train()
        labels = input[Conventions.LabelFrame]
        if labels.shape[1] == 1:
            return self._train_1_dim(input, labels)
        else:
            return self._train_multi_dim(input, labels)
