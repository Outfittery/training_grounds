from typing import *
from ... import batched_training as bt
import pandas as pd
from .conventions import Conventions
import torch
from yo_fluq_ds import Obj


class MulticlassPredictionInterpreter:
    def interpret(self, input, labels, output):
        result = input['index'].copy()
        for i, c in enumerate(labels.columns):
            result['true_' + c] = labels[c]
            result['predicted_' + c] = output[:, i].tolist()
        return result


class TorchModelHandler(bt.BatchedModelHandler):
    def __init__(self,
                 network_factory: Callable,
                 optimizer_factory: Callable,
                 loss_factory: Callable,
                 ignore_consistancy_check: bool
                 ):
        self.network_factory = network_factory
        self.optimizer_factory = optimizer_factory
        self.loss_factory = loss_factory
        self.multiclass_prediction_interpreter = MulticlassPredictionInterpreter()
        self.ignore_consistance_check = ignore_consistancy_check


    def instantiate(self, task, input: bt.IndexedDataBundle) -> None:
        self.network = self.network_factory(input)
        self.optimizer = self.optimizer_factory(self.network.parameters())
        self.loss = self.loss_factory()
        ldf = input[Conventions.LabelFrame]

        is_classification = True
        for c in ldf.columns:
            if len(ldf[c].unique())>2:
                is_classification = False
                break
        if is_classification and isinstance(self.loss,torch.nn.MSELoss) and not self.ignore_consistance_check:
            raise ValueError('Task seems to be classification task, but the metric is MSELoss. If it is intentional, set `ignore_consistancy_check` to True')


    def _predict_1_dim(self, input, labels):
        output = self.network(input)
        output = output.flatten().tolist()
        result = input['index'].copy()
        result['true'] = labels[labels.columns[0]]
        result['predicted'] = output
        return result

    def _predict_multi_dim(self, input, labels):
        output = self.network(input)
        result = self.multiclass_prediction_interpreter.interpret(input, labels, output)
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
