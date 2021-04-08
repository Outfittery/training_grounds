from typing import *

import pandas as pd
import logging

from ._hyperparams import _apply_hyperparams



logger = logging.getLogger(__name__)


class TrainingEnvironment:
    """
    This class isolates the training from the environment
    """

    def log(self, message) -> None:
        """Logs message"""
        pass

    def store_artifact(self, path: List[Any], name: Any, object: Any) -> None:
        """
        Saves artifact in a filesystem or memory.
        Args:
            path: path to the object from the filesystem/memory root
            name: name of the object (e.g. filename)
            object: the object
        """
        pass

    def output_metric(self, metric_name: str, metric_value: float) -> None:
        """
        Outputs the metric in the right format
        """
        pass

    def supports_tqdm(self) -> bool:
        """
        Returns: True if environment supports the tqdm (only local support it)
        """
        return True

    def flush(self) -> None:
        pass


class InMemoryTrainingEnvironment(TrainingEnvironment):
    """
    Default environment when the process is launched within current process, usually for debugging
    """

    def __init__(self):
        self.result = {'metrics': {}, 'runs': {}}

    def store_artifact(self, path: List[Any], name: Any, object: Any):
        loc = self.result
        for item in path:
            if item not in loc:
                loc[item] = {}
            loc = loc[item]
        loc[name] = object

    def output_metric(self, metric_name: str, metric_value: float):
        self.result['metrics'][metric_name] = metric_value

    def log(self, s):
        logger.info(s)


class AbstractTrainingTask:
    """Abstract class for training. This class is used as an integration point with, e.g., Sagemaker"""

    def __init__(self):
        self.info = {}

    def run_with_environment(self, data, env: Optional[TrainingEnvironment] = None):
        raise NotImplementedError()

    def get_metric_names(self):
        raise NotImplementedError()

    def run(self, data: Any):
        env = InMemoryTrainingEnvironment()
        self.run_with_environment(data, env)
        return env.result

    def apply_hyperparams(self, params: Dict[str, Any]):
        _apply_hyperparams(params, self)


class TrainingResult:
    def __init__(self):
        self.result_df = None  # type: Optional[pd.DataFrame]
        self.metrics = None  # type: Optional[Dict]
        self.info = None  # type: Optional[Dict]
        self.model = None  # Any
        self.training_task = None  # type: Optional[AbstractTrainingTask]
        self.train_split = None  # type: Optional[pd.Index]
        self.test_splits = None  # type: Optional[Dict[str,pd.Index]]


class ArtificierArguments:
    def __init__(self, result: TrainingResult, source_data: Any):
        self.result = result
        self.source_data = source_data


class Artificier:
    def run(self, args: ArtificierArguments):
        raise NotImplementedError()
