from .....common.ml.batched_training import *
from .....common.ml.dft import *
from .....common.ml.batched_training import torch as btorch

from sklearn import datasets
import pandas as pd
import torch.nn
import torch.optim
from sklearn.metrics import roc_auc_score

from typing import *

import torch
import pandas as pd

from torch.nn.modules.loss import _Loss

class TorchClassificationModelHandler(BatchedModelHandler):
    def __init__(self,
                 label_frame: str,
                 network_factory: Callable[[AbstractTrainingTask, Dict[str, pd.DataFrame]], torch.nn.Module],
                 optimizer_factory: Callable[[Any], torch.optim.Optimizer],
                 loss: _Loss = torch.nn.MSELoss()
                 ):
        self.label_frame = label_frame
        self.network_factory = network_factory
        self.optimizer_factory = optimizer_factory
        self.loss = loss
        self.network = None #type: Optional[torch.nn.Module]
        self.optimizer = None #type: Optional[torch.optim.Optimizer]

    def instantiate(self, task, input):
        self.network = self.network_factory(task, input)
        self.optimizer = self.optimizer_factory(self.network.parameters())

    def _get_xy(self, input):
        return {key: value for key, value in input.items() if key != self.label_frame}, input[self.label_frame]

    def train(self, input: Dict[str, pd.DataFrame]):
        X, y = self._get_xy(input)
        self.optimizer.zero_grad()
        output = self.network(X)
        targets = torch.tensor(y.values).float()
        loss = self.loss(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, input: Dict[str, pd.DataFrame]):
        X, y = self._get_xy(input)
        output = self.network(X)
        output = output.flatten().tolist()
        df = pd.DataFrame(dict(predicted=output, true=y[y.columns[0]]))
        return df

def create_bundle():
    iris = datasets.load_iris()
    df_features = pd.DataFrame(iris['data'], columns=['f1', 'f2', 'f3', 'f4'])
    df_targets = pd.DataFrame(iris['target'], columns=['label'])
    df_targets.label = df_targets.label==1
    index_df = pd.DataFrame(dict(features=df_features.index, targets=df_targets.index))
    bundle = DataBundle(index = index_df, features=df_features, targets=df_targets)
    return bundle



def create_task():
    feature_transformer = DataFrameTransformer([ContinousTransformer(['f1', 'f2', 'f3', 'f4'])])
    batcher = Batcher(10000,
                      [
                          PlainExtractor.build('features').index('features').apply(feature_transformer),
                          PlainExtractor.build('targets').index('targets').apply()
                      ],
                      PriorityRandomBatcherStrategy())

    splitter = CompositionSplitter(FoldSplitter(), FoldSplitter(decorate=True, test_name='display'))

    model_handler = TorchClassificationModelHandler(
        'targets',
        lambda task, input: btorch.FullyConnectedNetwork.Factory([100,1]).prepend_extraction('features').create_network(task, input),
        lambda parameters: torch.optim.SGD(parameters, lr=1)
    )

    task = BatchedTrainingTask(
        splitter=splitter,
        batcher=batcher,
        model_handler=model_handler,
        metric_pool=MetricPool().add_sklearn(roc_auc_score),
        settings=TrainingSettings(epoch_count=10),
    )
    return task