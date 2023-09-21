from typing import *
from ..._common import Logger
from ...ml.training_core import TrainingEnvironment

import pandas as pd
from pathlib import Path
import os
from yo_fluq_ds import FileIO
import subprocess


class SagemakerEnvironment(TrainingEnvironment):
    def __init__(self, model_folder: Path, checkpoint_path: Optional[Path] = None):
        self.model_folder = model_folder
        self.checkpoint_path = checkpoint_path

    def get_file_name(self, filename) -> Path:
        path = self.model_folder / filename
        os.makedirs(path.parent, exist_ok=True)
        return path

    def store_artifact(self, path: List[Any], name: Any, object: Any):
        op = self._store_artifact(self.model_folder, path, name, object)
        Logger.info(f"Saved artifact {op}")
        if self.checkpoint_path is not None:
            op = self._store_artifact(self.checkpoint_path, path, name, object)
            Logger.info(f"Saved artifact {op}")

    def _store_artifact(self, folder: Path, path: List[Any], name: Any, object: Any):
        output_path = folder
        for path_item in path:
            output_path /= str(path_item)
        os.makedirs(str(output_path), exist_ok=True)
        output_path /= str(name)
        if isinstance(object, pd.DataFrame):
            object.to_parquet(str(output_path) + ".parquet")
        else:
            FileIO.write_pickle(object, str(output_path) + '.pkl')
        return output_path


    def output_metric(self, metric_name: str, metric_value: float):
        metrics = f"###METRIC###{metric_name}:{metric_value}###"
        Logger.info(metrics)

    def supports_tqdm(self):
        return False


