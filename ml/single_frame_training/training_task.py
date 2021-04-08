from typing import *

import pandas as pd

from yo_ds import Query, fluq
from pathlib import Path

from ..training_core import AbstractTrainingTask, TrainingEnvironment, Splitter, IdentitySplit, Artificier, MetricPool, DataFrameSplit, TrainingResult, ArtificierArguments
from .df_loader import DataFrameLoader
from .model_provider import AbstractModelProvider
from ._kraken import _make_kraken_task



class Evaluation:
    @staticmethod
    def regression(model, X, y):
        return pd.DataFrame(dict(
            predicted=model.predict(X),
            true=y
        ))

    @staticmethod
    def binary_classification(model, X, y):
        return pd.DataFrame(dict(
            predicted=model.predict_proba(X)[:, 1],
            true=y
        ))

    @staticmethod
    def multiclass_classification(model, X, y):
        return pd.DataFrame(dict(
            predicted=model.predict(X),
            true=y
        ))


class SingleFrameTrainingTask(AbstractTrainingTask):
    """
    Class that describes the training process in case when all the training data is a single DataFrame,
    that fits to the memory
    """

    def __init__(self,
                 data_loader: DataFrameLoader,
                 model_provider: Optional[AbstractModelProvider] = None,
                 evaluator: Optional[Callable] = None,
                 metrics_pool: Optional[MetricPool] = None,
                 splitter: Splitter = IdentitySplit(),
                 artificers: Optional[List[Artificier]] = None,
                 index_column_name: str = 'original_index',
                 with_tqdm: bool = True,
                 late_initialization: Callable[['SingleFrameTrainingTask', DataFrameSplit, TrainingEnvironment], None] = None
                 ):
        super(SingleFrameTrainingTask, self).__init__()
        self.data_loader = data_loader
        self.model_provider = model_provider
        self.evaluator = evaluator
        self.splitter = splitter
        self.metrics_pool = metrics_pool
        self.index_column_name = index_column_name
        self.with_tqdm = with_tqdm
        self.artificers = artificers
        self.late_initialization = late_initialization

    def _iteration(self, dfs: DataFrameSplit, model_instance) -> TrainingResult:
        X, y = dfs.get_xy(dfs.train)
        model_instance.fit(X, y)
        result = []
        all_stages = {key: value for key, value in dfs.tests.items()}
        all_stages[DataFrameSplit.TRAIN_NAME] = dfs.train
        for key, value in all_stages.items():
            X, y = dfs.get_xy(value)
            r = self.evaluator(model_instance, X, y)
            r['stage'] = key

            if self.index_column_name is not None:
                r[self.index_column_name] = X.index

            result.append(r)

        result_df = pd.concat(result)
        result = TrainingResult()
        result.result_df = result_df
        result.info = {'split': dfs.info, **self.info}
        result.model = model_instance
        result.training_task = self
        result.train_split = dfs.train
        result.test_splits = dfs.tests

        all_artificiers = ([] if self.metrics_pool is None else [self.metrics_pool]) + ([] if self.artificers is None else self.artificers)

        args = ArtificierArguments(result,dfs)
        for artificer in all_artificiers:
            artificer.run(args)

        return result

    def _iteration_on_dfs(self, iteration: int, df: DataFrameSplit) -> TrainingResult:
        model_instance = self.model_provider.get_model(df)
        iteration_result = self._iteration(df, model_instance)
        iteration_result.info['iteration'] = iteration
        return iteration_result


    def _average_metrics(self, metrics_base):
        df = pd.DataFrame(metrics_base)
        metrics = Query.series(df.mean(axis=0)).to_dictionary()
        return metrics




    def _get_splits(self, data, env=None):
        dfs = self.data_loader.get_data(data)
        if self.late_initialization is not None:
            self.late_initialization(self, dfs, env)
        if self.model_provider is None:
            raise ValueError('model_provider is not set in constructor or in late initialization')

        if self.evaluator is None:
            raise ValueError('evaluator is not set in constructor or in late initialization')

        dfs = self.splitter(dfs)
        return dfs

    def run_with_environment(self, data: Union[pd.DataFrame, str, Path], env: Optional[TrainingEnvironment] = None):
        """
        Runs the training

        """
        if env is None:
            env = TrainingEnvironment()

        dfs = self._get_splits(data, env)


        stages_count = len(dfs)
        if self.with_tqdm and env.supports_tqdm():
            dfs = Query.en(dfs).feed(fluq.with_progress_bar())

        result = []
        metrics_base = []
        for i, df in enumerate(dfs):
            env.log(f'Starting stage {i + 1}/{stages_count}')
            iteration_result = self._iteration_on_dfs(i, df)
            if iteration_result.metrics is not None:
                metrics_base.append(iteration_result.metrics)
            for key, value in iteration_result.__dict__.items():
                env.store_artifact(['runs', i], key, value)
            result.append(iteration_result)
            env.log(f'Completed stage {i + 1}/{stages_count}')
            env.flush()

        if len(metrics_base) > 0:
            metrics = self._average_metrics(metrics_base)
        else:
            metrics = {}
        for key, value in metrics.items():
            env.output_metric(key, value)

    def get_metric_names(self):
        expected_stages = self.splitter.get_subset_names()
        if self.metrics_pool is None:
            return []
        metrics = [metric + '_' + stage for metric in self.metrics_pool.get_metrics_names() for stage in expected_stages]
        return metrics


    def make_kraken_task(self, configs: List[Any], data) -> Tuple[Callable, List[Any]]:
        return _make_kraken_task(self, configs, data)
