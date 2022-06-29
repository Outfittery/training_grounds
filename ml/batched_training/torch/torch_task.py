from typing import *

import pandas as pd
import torch

from yo_fluq_ds import Query

from ... import batched_training as bt
from ...single_frame_training import ModelConstructor
from .basis_tasks import AbstractBasisTaskSource
from .networks.network_commons import TorchNetworkFactory
from ...._common import Logger


class Conventions:
    LabelFrame = 'label'
    SplitColumnName = 'split'
    PriorityColumnName = 'priority'
    TrainName = 'train'
    DisplayName = 'display'
    TestName = 'test'


class OptimizerConstructor:
    def __init__(self, type_name, **kwargs):
        self.type_name = type_name
        self.kwargs = kwargs

    def instantiate(self, params):
        cls = ModelConstructor._load_class(self.type_name)
        return cls(params, **self.kwargs)


class TorchTrainingSettings:
    def __init__(self,
                 optimizer_ctor: Optional[OptimizerConstructor] = None,
                 loss_ctor: Optional[ModelConstructor] = None,
                 ):
        self.optimizer_ctor = optimizer_ctor if optimizer_ctor is not None else OptimizerConstructor('torch.optim:SGD', lr=1)
        self.loss_ctor = loss_ctor if loss_ctor is not None else ModelConstructor('torch.nn:MSELoss')


class TorchExtractorFactory:
    def create_extractors(self, task, bundle) -> List[bt.Extractor]:
        raise NotImplementedError()

    def preprocess_bundle(self, bundle):
        pass


class PredefinedExtractorFactory(TorchExtractorFactory):
    def __init__(self, *extractors):
        self.extractors = list(extractors)

    def create_extractors(self, task, bundle) -> List[bt.Extractor]:
        return self.extractors


class _TorchNetworkHandler(bt.BatchedModelHandler):
    def instantiate(self, task: 'TorchTrainingTask', input: Dict[str, pd.DataFrame]) -> None:
        self.network = task.network_factory.create_network(task, input)
        self.optimizer = task.torch_settings.optimizer_ctor.instantiate(self.network.parameters())
        self.loss = task.torch_settings.loss_ctor()

    def predict(self, input: Dict[str, pd.DataFrame]):
        labels = input[Conventions.LabelFrame]
        if labels.shape[1] != 1:
            raise ValueError(
                "TorchTrainingtask can be updated to handle multi-column labels, but it wasn't yet")  # TODO: add and test logic for multicolumn
        output = self.network(input)
        output = output.flatten().tolist()
        result = input['index'].copy()
        result['true'] = labels[labels.columns[0]]
        result['predicted'] = output
        return result

    def train(self, input: Dict[str, pd.DataFrame]) -> float:
        # TODO: add and test logic for multicolumn
        self.optimizer.zero_grad()
        result = self.network(input).flatten()
        target = torch.tensor(input[Conventions.LabelFrame].values).float().flatten()
        loss = self.loss(result, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class _BasisExtractorInfo:
    def __init__(self, extractor: bt.Extractor, basis_task_name: str):
        self.extractor = extractor
        self.basis_task_name = basis_task_name


class TorchTrainingTask(bt.BatchedTrainingTask):
    def __init__(self,
                 settings: bt.TrainingSettings,
                 torch_settings: TorchTrainingSettings,
                 extractor_factory: TorchExtractorFactory,
                 network_factory: TorchNetworkFactory,
                 metric_pool: bt.MetricPool,
                 basis_tasks_sources: Optional[Dict[str, AbstractBasisTaskSource]] = None
                 ):
        super(TorchTrainingTask, self).__init__(
            settings=settings,
            late_initialization=TorchTrainingTask.init,
            metric_pool=metric_pool,
            model_handler=_TorchNetworkHandler(),
            splitter=bt.PredefinedSplitter(
                Conventions.SplitColumnName,
                [Conventions.TestName, Conventions.DisplayName],
                [Conventions.TrainName, Conventions.DisplayName]
            )
        )
        self.torch_settings = torch_settings
        self.extractor_factory = extractor_factory
        self.network_factory = network_factory
        self.basis_tasks_sources = basis_tasks_sources
        self.basis_tasks = {}

    def _add_name_part(self, value, array):
        if value is not None:
            array.append(value)

    def init(self, bundle: bt.DataBundle):
        if self.basis_tasks_sources is not None:
            Logger.info('Loading basis tasks')
            for key, source in self.basis_tasks_sources.items():
                Logger.info(f'Loading basis task {key}')
                self.basis_tasks = source.load_task()
            Logger.info('Loading basis tasks finished')
        else:
            Logger.info('No basis tasks are available')

        Logger.info('Preprocessing bundle')
        self.extractor_factory.preprocess_bundle(bundle)

        if not self.settings.continue_training:
            Logger.info('Creating extractors')
            extractors = self.extractor_factory.create_extractors(self, bundle)
            Logger.info('Extractors: ' + ', '.join([i.get_name() for i in extractors]))

            if 'priority' in bundle.index_frame.columns:
                strategy = bt.PriorityRandomBatcherStrategy('priority')
            else:
                strategy = None

            Logger.info('Setting batcher')
            self.batcher = bt.Batcher(
                self.settings.batch_size,
                extractors,
                strategy)
