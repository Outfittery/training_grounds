from typing import *
from unittest import TestCase
from sklearn import datasets
from tg.common.ml import batched_training as bt
from tg.common.ml import dft
from tg.common.ml.batched_training import torch as btt
import pandas as pd
from yo_fluq_ds import fluq

pd.options.display.width=None

class TorchMultiColumnTestCase(TestCase):
    def test_iris(self):
        iris = datasets.load_iris()
        print(list(iris))
        features = pd.DataFrame(iris['data'], columns = iris['feature_names'])
        df = pd.DataFrame(iris['target_names'][iris['target']], columns = ['label'])
        df['split'] = bt.train_display_test_split(df, 0.2, 0.2, 'label')
        bundle = bt.DataBundle(index=df, features=features)
        task = btt.TorchTrainingTask(
            bt.TrainingSettings(
                epoch_count=100,
                batch_size=1000
            ),
            btt.TorchTrainingSettings(
                btt.OptimizerConstructor('torch.optim:SGD', lr=10)
            ),
            btt.PredefinedExtractorFactory(
                bt.PlainExtractor.build('features').index('features').apply(
                    transformer=dft.DataFrameTransformerFactory().default_factory()),
                bt.PlainExtractor.build('label').index().apply(
                    transformer=dft.DataFrameTransformerFactory().default_factory(),
                    take_columns='label'
                )
            ),
            btt.FullyConnectedNetwork.Factory([30], 'label').prepend_extraction('features'),
            bt.MetricPool().add(bt.RecallAtK(1)).add(bt.RecallAtK(2))
        )
        task.artificiers.append(bt.MulticlassWinningArtificier())
        result = task.run(bundle)
        self.assertGreater(result['metrics']['recall_at_1_test'], 0.8)
        self.assertGreater(result['metrics']['recall_at_2_test'], 0.9)