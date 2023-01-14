from typing import *

import copy
import os
import time

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from pathlib import Path

from .batcher import Batcher
from .data_bundle import DataBundle, IndexedDataBundle
from .model_handler import BatchedModelHandler
from ..training_core import AbstractTrainingTask, TrainingEnvironment, Splitter, MetricPool, DataFrameSplit, TrainingResult, ArtificierArguments, IdentitySplitter
from ..._common import Logger


class TrainingSettings:
    """
    Settings of the training process
    """

    def __init__(self,
                 epoch_count: int = 100,
                 batch_size: int = 10000,
                 continue_training: bool = False,
                 training_time_limit: Optional[timedelta] = None,
                 evaluation_time_limit: Optional[timedelta] = None,
                 training_batch_limit: Optional[int] = None,
                 evaluation_batch_limit: Optional[int] = None,
                 is_quite_time: Optional[Callable[[datetime], bool]] = None,
                 mini_batch_size: Optional[int] = None,
                 mini_epoch_count: Optional[int] = None,
                 mini_reporting_conventional: bool = True,
                 delay_after_iteration_in_seconds: Optional[float] = None,
                 index_frame_name_in_bundle: str = 'index'
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
        self.mini_batch_size = mini_batch_size
        self.mini_epoch_count = mini_epoch_count
        self.batch_size = batch_size
        self.mini_reporting_conventional = mini_reporting_conventional
        self.delay_after_iteration_in_seconds = delay_after_iteration_in_seconds
        self.index_frame_name_in_bundle = index_frame_name_in_bundle

    def mini_batches_are_requried(self):
        return self.mini_batch_size is not None


class _TrainingTempData:
    def __init__(self, ibundle: IndexedDataBundle, env: TrainingEnvironment, split: DataFrameSplit, first_iteration: int):
        self.original_ibundle = ibundle
        self.env = env
        self.split = split
        self.first_iteration = first_iteration
        self.iteration = 0
        self.losses = []
        self.epoch_begins_at = None  # type: Optional[datetime]
        self.train_bundle = None  # type: Optional[IndexedDataBundle]
        self.result = None  # type: Optional[TrainingResult]
        self.batch = None  # type: Dict[str, pd.DataFrame]
        self.mini_batch_indices = None  # type: List
        self.mini_batch = None  # type: Dict[str, pd.DataFrame]


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
                 late_initialization: Callable[['BatchedTrainingTask', DataBundle], None] = None,
                 debug=False
                 ):
        super(BatchedTrainingTask, self).__init__()
        self.batcher = batcher
        self.model_handler = model_handler
        self.metric_pool = metric_pool
        self.splitter = splitter or IdentitySplitter()
        self.settings = settings or TrainingSettings()
        self.history = None
        self.artificiers = artificiers or []
        self.late_initialization = late_initialization
        self.debug = debug

    # region Initialization

    def _get_split(self, ibundle: IndexedDataBundle):
        initial_split = DataFrameSplit(ibundle.index_frame, [], None)
        split = self.splitter(initial_split)
        if len(split) != 1:
            raise ValueError('Splitter must provide exactly one split in case of batch training')
        split = split[0]
        return split

    def _instantiate_all(self, ibundle: IndexedDataBundle):
        self.history = []

        Logger.info('Fitting the transformers')
        test_batch = self.batcher.fit_extract(self.settings.batch_size, ibundle)

        Logger.info('Instantiating model')
        self.model_handler.instantiate(self, test_batch)

    @staticmethod
    def _ensure_bundle(inp: Union[Path, str, DataBundle, IndexedDataBundle], index_frame_name) -> IndexedDataBundle:
        if isinstance(inp, IndexedDataBundle):
            return inp

        if isinstance(inp, DataBundle):
            bundle = inp
        elif isinstance(inp, str) or isinstance(inp, Path):
            inp = Path(inp)
            if os.path.isdir(inp):
                bundle = DataBundle._read_bundle(inp)
            elif os.path.isfile(inp):
                bundle = DataBundle._load_zip(inp)
            else:
                raise ValueError(f'Path {inp} not found')
        else:
            raise ValueError(
                f'`inp` was expected to be DataBundle or the path to folder that contains Data Bundle parquet files, but was {type(inp)}')

        ibundle = IndexedDataBundle(bundle[index_frame_name], bundle)
        return ibundle

    def _prepare_all(self, bundle: Union[str, Path, DataBundle], env: Optional[TrainingEnvironment]) -> _TrainingTempData:
        if env is None:
            env = TrainingEnvironment()
        Logger.info(f'Training starts. Info: {self.info}')

        Logger.info(f'Ensuring/loading bundle. Bundle before:\n{bundle}')
        ibundle = BatchedTrainingTask._ensure_bundle(bundle, self.settings.index_frame_name_in_bundle)
        Logger.info(f'Bundle loaded\n{ibundle.bundle.describe(5)}')
        Logger.info(f'Index frame is set to {self.settings.index_frame_name_in_bundle}, shape is {ibundle.index_frame.shape}')

        if self.late_initialization is not None:
            Logger.info('Running late initizalization')
            self.late_initialization(self, ibundle)
        else:
            Logger.info('Skipping late initialization')

        if self.model_handler is None:
            raise ValueError('model_handler was not set up in late_initialization nor in constructor')

        if self.batcher is None:
            raise ValueError('batcher was not set up in late_initialization nor in constructor')

        Logger.info('Preprocessing bundle by batcher')
        self.batcher.preprocess_bundle(ibundle)

        split = self._get_split(ibundle)
        split_msg = f'Splits: train {len(split.train)}, ' + ', '.join([f"{key} {len(value)}" for key, value in split.tests.items()])
        Logger.info(split_msg)

        if not self.settings.continue_training:
            Logger.info('New training. Instantiating the system')
            self._instantiate_all(ibundle.change_index(split.train))
            first_iteration = 0
        else:
            Logger.info('Continued training.')
            first_iteration = len(self.history)
        return _TrainingTempData(ibundle, env, split, first_iteration)

    def generate_sample_batch_and_temp_data(self, bundle: DataBundle, batch_index: int = 0, from_split=None, force_default_strategy=False):
        temp_data = self._prepare_all(bundle, None)
        if from_split is None:
            ibundle = temp_data.original_ibundle.change_index(temp_data.split.train)
        else:
            ibundle = temp_data.original_ibundle.change_index(temp_data.split.tests[from_split])
        batch = self.batcher.get_batch(self.settings.batch_size, ibundle, batch_index, force_default_strategy)
        return batch, temp_data

    @staticmethod
    def _get_metric_names(expected_stages: List[str], metric_pool: MetricPool):
        if metric_pool is not None:
            metrics = [metric_name + '_' + stage for metric_name in metric_pool.get_metrics_names() for stage in
                       expected_stages]
        else:
            metrics = []

        metrics.append('loss')
        metrics.append('iteration')
        return metrics


    def get_metric_names(self):
        return BatchedTrainingTask._get_metric_names(
            self.splitter.get_subset_names(),
            self.metric_pool
        )


    # endregion

    # region Prediction

    def _evaluation_for_one_stage(self, ibundle: IndexedDataBundle, stage_name: str):
        dfs = []
        batch_count = self.batcher.get_batch_count(self.settings.batch_size, ibundle, True)
        if batch_count == 0:
            raise ValueError('There is no batches!')
        evaluation_begin = datetime.now()
        for i in range(0, batch_count):
            self._wait_till_end_of_quite_hours()
            iteration_begin = datetime.now()
            if self.settings.evaluation_time_limit is not None and iteration_begin - evaluation_begin > self.settings.evaluation_time_limit:
                break
            if self.settings.evaluation_batch_limit is not None and i >= self.settings.evaluation_batch_limit:
                break
            Logger.info(f"{stage_name}: {i}/{batch_count}")
            batch = self.batcher.get_batch(self.settings.batch_size, ibundle, i, True)
            df_addition = self.model_handler.predict(batch)
            dfs.append(df_addition)
        df = pd.concat(dfs, sort=False)
        return df

    def _evaluation_df(self, ibundle: IndexedDataBundle, split: DataFrameSplit):
        dfs = []
        for stage_name, stage_index in split.tests.items():
            bundle_with_index = ibundle.change_index(stage_index)
            df = self._evaluation_for_one_stage(bundle_with_index, stage_name)
            df['stage'] = stage_name
            dfs.append(df)
        if len(dfs)>0:
            return pd.concat(dfs, sort=False)
        return pd.DataFrame([])

    def predict(self, ibundle: Union[str, Path, DataBundle, IndexedDataBundle]):
        ibundle = self._ensure_bundle(ibundle, self.settings.index_frame_name_in_bundle)
        self.batcher.preprocess_bundle(ibundle)
        return self._evaluation_for_one_stage(ibundle, 'prediction')

    # endregion

    # region Training

    def _wait_till_end_of_quite_hours(self):
        if self.settings.is_quite_time is not None:
            while self.settings.is_quite_time(datetime.now()):
                time.sleep(60)
                Logger.info("Quite time")

    def _training_report(self, temp_data: _TrainingTempData):
        result = TrainingResult()
        temp_data.result = result
        result.result_df = self._evaluation_df(temp_data.original_ibundle, temp_data.split)

        result.model = self.model_handler
        result.batcher = self.batcher
        result.training_task = self
        result.test_splits = temp_data.split.tests
        result.train_split = temp_data.split.train

        result.metrics = {}


        args = ArtificierArguments(result, temp_data.original_ibundle)

        for artificier in self.artificiers:
            artificier.run_before_metrics(args)

        if result.result_df.shape[0]>0 and self.metric_pool is not None:
            self.metric_pool.run(args)

        result.metrics['loss'] = np.mean(temp_data.losses)
        result.metrics['iteration'] = temp_data.first_iteration + temp_data.iteration

        for key, value in result.metrics.items():
            temp_data.env.output_metric(key, value)

        result.metrics['timestamp'] = str(datetime.now())
        self.history.append(result.metrics)
        result.history = copy.deepcopy(self.history)

        for artificier in self.artificiers:
            artificier.run_before_storage(args)

        for key, value in result.__dict__.items():
            temp_data.env.store_artifact(['output'], key, value)

        temp_data.env.flush()
        temp_data.iteration += 1
        temp_data.losses = []

    def _check_training_time_conditions(self, temp_data: _TrainingTempData, batch_number: int):
        self._wait_till_end_of_quite_hours()
        iteration_begin = datetime.now()
        if self.settings.training_time_limit is not None and iteration_begin - temp_data.epoch_begins_at > self.settings.training_time_limit:
            Logger.info('Interrupted because of the training_time_limit')
            return False
        if self.settings.training_batch_limit is not None and batch_number >= self.settings.training_batch_limit:
            Logger.info('Interrupted because of the training_batch_limit')
            return False
        return True

    def _train_simple_epoch(self, temp_data: _TrainingTempData):
        temp_data.losses = []
        temp_data.epoch_begins_at = datetime.now()
        batch_count = self.batcher.get_batch_count(self.settings.batch_size, temp_data.train_bundle)
        if batch_count == 0:
            raise ValueError('There is no batches!')
        for i in range(0, batch_count):
            if not self._check_training_time_conditions(temp_data, i):
                break
            Logger.info(f"Training: {i}/{batch_count}")
            batch = self.batcher.get_batch(self.settings.batch_size, temp_data.train_bundle, i)
            temp_data.batch = batch
            loss = self.model_handler.train(batch)
            temp_data.losses.append(loss)
        self._training_report(temp_data)

    def _train_epoch_with_minibatches(self, temp_data: _TrainingTempData):
        temp_data.losses = []
        temp_data.epoch_begins_at = datetime.now()
        batch_count = self.batcher.get_batch_count(self.settings.batch_size, temp_data.train_bundle)
        if batch_count == 0:
            raise ValueError('There is no batches!')
        terminate = False

        for i in range(0, batch_count):
            if terminate:
                break
            batch = self.batcher.get_batch(self.settings.batch_size, temp_data.train_bundle, i)
            temp_data.batch = batch
            mini_epochs = self.settings.mini_epoch_count or 1
            for j in range(0, mini_epochs):
                if not self._check_training_time_conditions(temp_data, i):
                    terminate = True
                    break
                Logger.info(f"Training: {i}/{batch_count} batch, {j}/{mini_epochs} mini-epoch")
                mini_indices = self.batcher.get_mini_batch_indices(self.settings.mini_batch_size, batch)
                temp_data.mini_batch_indices = mini_indices
                for mini_index in mini_indices:
                    mini_batch = self.batcher.get_mini_batch(mini_index, batch)
                    temp_data.mini_batch = mini_batch
                    loss = self.model_handler.train(mini_batch)
                    temp_data.losses.append(loss)

                if not self.settings.mini_reporting_conventional:
                    self._training_report(temp_data)

        if self.settings.mini_reporting_conventional:
            self._training_report(temp_data)

    def run_with_environment(self, _bundle: Union[str, Path, DataBundle], env: Optional[TrainingEnvironment] = None):
        temp_data = self._prepare_all(_bundle, env)
        Logger.info('Initialization completed')

        temp_data.train_bundle = temp_data.original_ibundle.change_index(temp_data.split.train)
        if self.debug:
            self.data_ = temp_data

        for i in range(self.settings.epoch_count):
            Logger.info(f"Epoch {i} of {self.settings.epoch_count}")
            if not self.settings.mini_batches_are_requried():
                self._train_simple_epoch(temp_data)
            else:
                self._train_epoch_with_minibatches(temp_data)
            if self.settings.delay_after_iteration_in_seconds is not None:
                time.sleep(self.settings.delay_after_iteration_in_seconds)

    # endregion
