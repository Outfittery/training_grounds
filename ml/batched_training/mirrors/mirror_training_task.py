from typing import *

from .. import TrainingSettings, MetricPool
from .. import torch as btt
from .arch import MirrorSettings, _MirrorExtractorFactory, _MirrorNetworkFactory


class MirrorTrainingTask(btt.TorchTrainingTask):
    def __init__(self,
                 training_settings: TrainingSettings,
                 torch_settings: btt.TorchTrainingSettings,
                 mirror_settings: MirrorSettings,
                 metric_pool: MetricPool,
                 basis_tasks_sources: Optional[Dict[str, btt.AbstractBasisTaskSource]] = None
                 ):
        extractor_factory = _MirrorExtractorFactory(mirror_settings)
        network_factory = _MirrorNetworkFactory(mirror_settings)
        super(MirrorTrainingTask, self).__init__(
            training_settings,
            torch_settings,
            extractor_factory,
            network_factory,
            metric_pool,
            basis_tasks_sources
        )
