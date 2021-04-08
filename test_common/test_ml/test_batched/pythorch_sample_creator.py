from .....common.ml.batched_training import *
from .....common.ml.dft import *
from .....common.ml.batched_training import torch as btorch

from sklearn import datasets
import pandas as pd
import torch.nn
import torch.optim
from sklearn.metrics import roc_auc_score


def create_bundle():
    iris = datasets.load_iris()
    df_features = pd.DataFrame(iris['data'], columns=['f1', 'f2', 'f3', 'f4'])
    df_targets = pd.DataFrame(iris['target'], columns=['label'])
    df_targets.label = df_targets.label==1
    index_df = pd.DataFrame(dict(features=df_features.index, targets=df_targets.index))
    bundle = DataBundle(index_df, dict(features=df_features, targets=df_targets))
    return bundle

def create_model(task, input):
    network = btorch.MultiLayerPerceptronWithParsing(input, ['features'],[100,1])
    return btorch.TorchModel(
        network,
        torch.optim.SGD(network.parameters(), lr=1)
    )

def create_task():
    feature_transformer = DataFrameTransformer([ContinousTransformer(['f1', 'f2', 'f3', 'f4'])])
    batcher = Batcher(10000,
                      [
                          DirectExtractor('features', feature_transformer),
                          DirectExtractor('targets')
                      ],
                      PriorityRandomBatcherStrategy())

    splitter = CompositionSplit(FoldSplitter(), FoldSplitter(decorate=True, test_name='display'))

    model_handler = btorch.TorchClassificationModelHandler('targets', create_model)

    task = BatchedTrainingTask(
        splitter=splitter,
        batcher=batcher,
        model_handler=model_handler,
        metric_pool=MetricPool().add_sklearn(roc_auc_score),
        settings=TrainingSettings(epoch_count=10),
    )
    return task