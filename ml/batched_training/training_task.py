from typing import *

import copy
import time
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from pathlib import Path

from .batcher import Batcher
from .data_bundle import DataBundle, IndexedDataBundle
from .model_handler import BatchedModelHandler
from ..training_core import AbstractTrainingTask, TrainingEnvironment, Splitter, MetricPool, DataFrameSplit, TrainingResult, ArtificierArguments, IdentitySplit



class TrainingSettings:
    """
    Settings of the training process
    """

    def __init__(self,
                 epoch_count: int = 100,
                 continue_training: bool = False,
                 training_time_limit: Optional[timedelta] = None,
                 evaluation_time_limit: Optional[timedelta] = None,
                 training_batch_limit: Optional[int] = None,
                 evaluation_batch_limit: Optional[int] = None,
                 is_quite_time: Optional[Callable[[datetime], bool]] = None
                 ):
        """

        Args:
            epoch_count: for how much epochs the process should lasts
        """
        self.epoch_count = epoch_count
        self.continue_training = continue_training
        self.training_time_limit = training_time_limit
        self.training_batch_limit = training_batch_limit
        self.evaluation_batch_limit = evaluation_batch_limit
        self.evaluation_time_limit = evaluation_time_limit
        self.is_quite_time = is_quite_time


class BatchedTrainingTask(AbstractTrainingTask):
    """
    Training process for the case, when the training data does not fit to the memory.
    The data are produces batch after batch by batcher, the model is trained on each batch.
    After one epoch (i.e. all the batches) are used for training, the model is evaluated on
    test sets.
    """

    def __init__(self,
                 splitter: Optional[Splitter] = None,
                 batcher: Optional[Batcher] = None,
                 model_handler: Optional[BatchedModelHandler] = None,
                 metric_pool: Optional[MetricPool] = None,
                 settings: Optional[TrainingSettings] = None,
                 artificiers: Optional[List[Any]] = None,
                 late_initialization: Callable[['BatchedTrainingTask', DataBundle, TrainingEnvironment], None] = None
                 ):
        super(BatchedTrainingTask, self).__init__()
        self.batcher = batcher
        self.model_handler = model_handler
        self.metric_pool = metric_pool
        self.splitter = splitter or IdentitySplit()
        self.settings = settings or TrainingSettings()
        self.history = None
        self.artificiers = artificiers or []
        self.late_initialization = late_initialization

    def _wait_till_end_of_quite_hours(self, env: TrainingEnvironment):
        if self.settings.is_quite_time is not None:
            while self.settings.is_quite_time(datetime.now()):
                time.sleep(60)
                env.log("Quite time")

    def _training(self, bundle: IndexedDataBundle, env: TrainingEnvironment):
        losses = []
        batch_count = self.batcher.get_batch_count(bundle)
        if batch_count == 0:
            raise ValueError('There is no batches!')
        training_begin = datetime.now()
        for i in range(0, batch_count):
            self._wait_till_end_of_quite_hours(env)
            iteration_begin = datetime.now()
            if self.settings.training_time_limit is not None and iteration_begin - training_begin > self.settings.training_time_limit:
                break
            if self.settings.training_batch_limit is not None and i >= self.settings.training_batch_limit:
                break
            env.log(f"Training: {i}/{batch_count}")
            batch = self.batcher.get_batch(bundle, i)
            loss = self.model_handler.train(batch)
            losses.append(loss)
        return np.mean(losses)

    def _evaluation_for_one_stage(self, bundle: IndexedDataBundle, stage_name: str, env: TrainingEnvironment):
        dfs = []
        batch_count = self.batcher.get_batch_count(bundle, True)
        if batch_count == 0:
            raise ValueError('There is no batches!')
        evaluation_begin = datetime.now()
        for i in range(0, batch_count):
            self._wait_till_end_of_quite_hours(env)
            iteration_begin = datetime.now()
            if self.settings.evaluation_time_limit is not None and iteration_begin - evaluation_begin > self.settings.evaluation_time_limit:
                break
            if self.settings.evaluation_batch_limit is not None and i >= self.settings.evaluation_batch_limit:
                break
            env.log(f"{stage_name}: {i}/{batch_count}")
            batch = self.batcher.get_batch(bundle, i, True)
            df_addition = self.model_handler.predict(batch)
            dfs.append(df_addition)
        df = pd.concat(dfs, sort=False)
        return df

    def _evaluation_df(self, bundle: DataBundle, split: DataFrameSplit, env: TrainingEnvironment):
        dfs = []
        for stage_name, stage_index in split.tests.items():
            bundle_with_index = IndexedDataBundle(bundle, stage_index)
            df = self._evaluation_for_one_stage(bundle_with_index, stage_name, env)
            df['stage'] = stage_name
            dfs.append(df)
        return pd.concat(dfs, sort=False)

    def _get_split(self, bundle):
        initial_split = DataFrameSplit(bundle.index_frame, [], None)
        split = self.splitter(initial_split)
        if len(split) != 1:
            raise ValueError('Splitter must provide exactly one split in case of batch training')
        split = split[0]
        return split

    def _instantiate_all(self, bundle: IndexedDataBundle, env):
        self.history = []

        env.log('Fitting the transformers')
        self.batcher.fit_extractors(bundle.bundle)

        env.log('Splitting index')

        env.log('Creating sample batch')
        batch = self.batcher.get_batch(bundle, 0)

        env.log('Instantiating model')
        self.model_handler.instantiate(self, batch)

    def run_with_environment(self, bundle: Union[str, Path, DataBundle], env: Optional[TrainingEnvironment] = None):
        bundle = DataBundle.ensure(bundle)

        if env is None:
            env = TrainingEnvironment()

        if self.late_initialization is not None:
            self.late_initialization(self, bundle, env)

        if self.model_handler is None:
            raise ValueError('model_handler was not set up in late_initialization nor in constructor')

        if self.batcher is None:
            raise ValueError('batcher was not set up in late_initialization nor in constructor')

        split = self._get_split(bundle)

        if not self.settings.continue_training:
            self._instantiate_all(bundle.as_indexed(split.train), env)
            first_iteration = 0
        else:
            first_iteration = len(self.history)

        for i in range(self.settings.epoch_count):
            env.log(f"Epoch {i} of {self.settings.epoch_count}")
            loss = self._training(IndexedDataBundle(bundle, split.train), env)
            result = TrainingResult()
            result.result_df = self._evaluation_df(bundle, split, env)

            result.model = self.model_handler
            result.batcher = self.batcher
            result.training_task = self
            result.test_splits = split.tests
            result.train_split = split.train

            args = ArtificierArguments(result, bundle)
            if self.metric_pool is not None:
                self.metric_pool.run(args)
            else:
                result.metrics = {}

            result.metrics['loss'] = loss
            result.metrics['iteration'] = first_iteration + i
            self.history.append(result.metrics)
            result.history = copy.deepcopy(self.history)

            for artificier in self.artificiers:
                artificier.run(args)

            for key, value in result.__dict__.items():
                env.store_artifact(['output'], key, value)

            for key, value in result.metrics.items():
                env.output_metric(key, value)

            env.flush()

    def get_metric_names(self):
        expected_stages = self.splitter.get_subset_names()
        if self.metric_pool is not None:
            metrics = [metric_name + '_' + stage for metric_name in self.metric_pool.get_metrics_names() for stage in
                       expected_stages]
        else:
            metrics = []

        metrics.append('loss')
        metrics.append('iteration')
        return metrics
