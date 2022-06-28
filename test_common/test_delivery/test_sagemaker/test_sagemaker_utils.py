from unittest import TestCase
from sklearn.metrics import roc_auc_score, f1_score
from tg.common.delivery.training.sagemaker_training_routine import get_sagemaker_metric_definitions
from tg.common.ml.single_frame_training import *
from yo_fluq_ds import *


class SagemakerUtilsTest(TestCase):
    def test_metric_definictions(self):
        task = SingleFrameTrainingTask(
            DataFrameLoader('label'),
            ModelProvider(ModelConstructor('')),
            Evaluation.binary_classification,
            MetricPool().add_sklearn(roc_auc_score).add_sklearn(f1_score),
            CompositionSplitter(
                FoldSplitter(10, test_name='test'),
                FoldSplitter(1, test_name='validation')
            ),
        )
        metrics = get_sagemaker_metric_definitions(task)
        names = Query.en(metrics).select(lambda z: z['Name']).order_by(lambda z: z).to_list()
        self.assertListEqual(
            ['f1_score_test', 'f1_score_train', 'f1_score_validation', 'roc_auc_score_test', 'roc_auc_score_train', 'roc_auc_score_validation'],
            names
        )
