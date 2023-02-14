from typing import *
from ... import batched_training as bt
from .conventions import Conventions
from .torch_model_handler import TorchModelHandler
from .networks.basics import CtorAdapter

class AssemblyPoint:
    def create_network_factory(self):
        raise NotImplementedError()

    def create_extractor(self):
        raise NotImplementedError()


def _initialization_bridge(task: 'TorchTrainingTask', data: bt.IndexedDataBundle) -> None:
    task.fix_bundle(data)
    if not task.settings.continue_training:
        task.initialize_task(data)


class  TorchTrainingTask(bt.BatchedTrainingTask):
    def __init__(self):
        splitter = bt.PredefinedSplitter(
                Conventions.SplitColumnName,
                [Conventions.TestName, Conventions.DisplayName],
                [Conventions.TrainName, Conventions.DisplayName]
            )
        super(TorchTrainingTask, self).__init__(
            settings = bt.TrainingSettings(),
            splitter = splitter,
            metric_pool= bt.MetricPool(),
            late_initialization=_initialization_bridge
        )
        self.optimizer_ctor = CtorAdapter('torch.optim:SGD', ('params',), lr = 0.1)
        self.loss_ctor = CtorAdapter('torch.nn:MSELoss')
        self.settings.mini_batch_size = 200
        self.settings.mini_epoch_count = 4

    def fix_bundle(self, idb: bt.IndexedDataBundle):
        pass

    def initialize_task(self, idb: bt.IndexedDataBundle):
        raise NotImplementedError()

    def setup_batcher(self, ibundle, extractors, index_frame_name='index', stratify_by_column = None):
        strategy = None
        if stratify_by_column is not None:
            df = ibundle.bundle[index_frame_name]
            df[Conventions.PriorityColumnName] = bt.PriorityRandomSampler.make_priorities_for_even_representation(df, stratify_by_column)
            strategy = bt.PriorityRandomSampler(Conventions.PriorityColumnName)
        self.batcher = bt.Batcher(extractors, strategy)


    def setup_model(self, network_factory, ignore_consistancy_check = False):
        self.model_handler = TorchModelHandler(network_factory, self.optimizer_ctor, self.loss_ctor, ignore_consistancy_check)




