from sklearn import datasets
from unittest import TestCase
from sklearn.linear_model import LogisticRegression, Lasso
from yo_ds import kraken
from tg.common.ml.single_frame_training import *
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tg.common.test_common.test_ml.test_single_frame.task import get_data, create_task

def metric(result):
    accuracies = []
    for rs in result:
        df = rs['result_df']
        df = df.loc[df.stage == 'test']
        accuracies.append(accuracy_score(df.true, df.predicted))
    return np.median(accuracies)



DF = get_data()

class TaskTestCase(TestCase):
    def test_simple(self):
        result = create_task().run(DF)['runs']
        self.assertEqual(1, len(result))
        self.assertIsInstance(result[0]['result_df'], pd.DataFrame)
        self.assertIsInstance(result[0]['model'], LogisticRegression)

    def test_with_hypers(self):
        task = create_task()
        task.apply_hyperparams({
            'splitter.fold_count:int': '2',
            'model_provider.constructor.type_name': 'sklearn.linear_model:Lasso',
            'model_provider.constructor.kwargs.alpha:float': '2.5'
        })
        result = task.run(DF)['runs']
        self.assertEqual(2, len(result))
        self.assertIsInstance(result[0]['model'], Lasso)

    def test_with_kraken(self):
        task = create_task()
        pre_config = Query.combinatorics.grid_dict(
            {
                'splitter.fold_count:int': [2, 3],
                'model_provider.constructor.kwargs.C:float': [1, 2]
            })
        callable, config = task.make_kraken_task(pre_config, DF)
        result = kraken.release(callable, config, None)
        self.assertEqual(10, len(result))

    def test_with_metric(self):
        task = create_task()
        task.splitter.fold_count = 10
        task.metrics_pool = MetricPool().add_sklearn(accuracy_score).add_sklearn(f1_score, average='macro')
        result = task.run(DF)
        self.assertEqual(10, len(result['runs']))
        self.assertSetEqual({'accuracy_score_test', 'f1_score_test', 'accuracy_score_train', 'f1_score_train'}, set(result['metrics'].keys()))

    def test_with_wrapper(self):
        task = create_task()
        task.model_provider = ModelProvider(ModelConstructor('catboost:CatBoostClassifier'),None,ModelProvider.catboost_model_fix)
        task.model_provider.constructor.kwargs['silent'] = True
        task.model_provider.constructor.kwargs['iterations'] = 5
        tdf = DF.assign(target=np.where(DF.target==0,0,1))
        task.run(tdf)

