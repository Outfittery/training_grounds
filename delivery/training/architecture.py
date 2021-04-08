from typing import *

import datetime
import json
import os
import shutil
import subprocess
import jsonpickle

from pathlib import Path
from uuid import uuid4
from yo_fluq_ds import FileIO

from ..packaging import EntryPoint, HackedUnpicker
from ..._common import Loc
from ...ml.training_core import AbstractTrainingTask, TrainingEnvironment



_TRAINING_RESULTS_LOCATION = Loc.temp_path/'training_results'


class TrainingExecutor:
    def upload_dataset(self, dataset_version: str):
        raise NotImplementedError()

    def download_dataset(self, dataset_version: str):
        raise NotImplementedError()

    def upload_shared_data(self, source_path: Path, dst_key: str):
        raise NotImplementedError()

    def execute(self, task: AbstractTrainingTask, dataset_version: str, wait=True) -> str:
        raise NotImplementedError()

    def get_result(self, id: str):
        raise NotImplementedError()


class FileCacheTrainingEnvironment(TrainingEnvironment):
    def __init__(self, logging_method: Callable, model_folder: Path, zip_on_flush_to_file: Optional[Path] = None):
        self.logging_method = logging_method
        self.model_folder = model_folder
        self.zip_on_flush_to_file = zip_on_flush_to_file

    def log(self, message):
        self.logging_method(message)

    def get_file_name(self, filename) -> Path:
        path = self.model_folder/filename
        os.makedirs(path.parent, exist_ok=True)
        return path

    def store_artifact(self, path: List[Any], name: Any, object: Any):
        output_path = self.model_folder
        for path_item in path:
            output_path /= str(path_item)
        os.makedirs(str(output_path),exist_ok=True)
        output_path /= str(name)
        FileIO.write_pickle(object, output_path)
        self.log(f"Saved artifact {output_path}")

    def output_metric(self, metric_name: str, metric_value:float):
        metrics = f"###METRIC###{metric_name}:{metric_value}###"
        self.log(metrics)

    def supports_tqdm(self):
        return False

    def flush(self) -> None:
        if self.zip_on_flush_to_file is not None:
            self.log(f"Flushing result to {self.zip_on_flush_to_file}")
            if self.zip_on_flush_to_file.is_file():
                os.remove(str(self.zip_on_flush_to_file))
            tar_command = ['tar', '-czf', str(self.zip_on_flush_to_file), '-C', self.model_folder, '.', "--transform=s/^.//"]
            self.log(tar_command)
            subprocess.call(tar_command)
            subprocess.call(['chmod', '777', str(self.zip_on_flush_to_file)])

    def common_initialization(self, entry: EntryPoint, package_location: Optional[Path]=None, hyperparams:Optional[Dict]=None, resource_name='model') -> AbstractTrainingTask:
        self.log('Common Training Initialization')

        self.log('Preparing package properties...')
        package_props = entry.get_properties()
        props_str = json.dumps(package_props)
        self.log(props_str)
        FileIO.write_text(props_str, self.get_file_name('package.json'))

        if package_location is not None:
            self.log('Preparing package file...')
            shutil.copy(str(package_location), str(self.get_file_name('package.tag.gz')))

        self.log('Loading model from package...')
        model = entry.load_resource(resource_name)

        if hyperparams is None:
            self.log("No hyperparameters are provided")
        else:
            self.log("Hyperparameters are provided")
            self.log(hyperparams)
            self.log("Storing Hyperparameters in file")
            FileIO.write_json(hyperparams, self.get_file_name('hyperparameters.json'))
            self.log(f'Applying hyperparams')
            model.apply_hyperparams(hyperparams)

        self.log("Model initialized. Jsonpickling...")
        model_state = json.dumps(json.loads(jsonpickle.dumps(model)), indent=1)
        FileIO.write_text(model_state, self.get_file_name('task.json'))

        self.log("Common Training Initialization completed")
        return model


class ResultPickleReader:
    def __init__(self, path: Path, hacked_unpickling = True):
        self.path = path
        self.hacked_unpickling = hacked_unpickling

    def unpickle(self, subpath) -> Any:
        if self.hacked_unpickling:
            package = FileIO.read_jsonpickle(self.path.joinpath('package.json'))
            with open(str(self.path.joinpath(subpath)), 'rb') as file:
                unpickler = HackedUnpicker(file, package['tg_module_name'], package['original_tg_module_name'])
                return unpickler.load()
        else:
            return FileIO.read_pickle(self.path.joinpath(subpath))

    def get_path(self, subpath) -> Path:
        return self.path.joinpath(subpath)


    @staticmethod
    def from_tar_gz_file(path_to_tar_gz: Path, id: str, path_to_unpacked_folder: Optional[Path] = None):
        if path_to_unpacked_folder is None:
            path_to_unpacked_folder = Loc.temp_path.joinpath('training_jobs_results')
        folder = path_to_unpacked_folder/id
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        tar_call = ['tar', '-C', folder, '-xvf', path_to_tar_gz]
        subprocess.call(tar_call)
        return ResultPickleReader(Path(folder))


class TrainingRoutineBase:
    def __init__(self, local_dataset_storage: Path):
        self.local_dataset_storage = local_dataset_storage


class AttachedTrainingExecutor(TrainingExecutor):
    def __init__(self, routine: TrainingRoutineBase):
        self.routine = routine
        self.print_function = print

    def execute(self, task: AbstractTrainingTask, dataset_version: str, wait=True) -> str:
        id = _create_id(task.info.get('name',''))
        folder = _TRAINING_RESULTS_LOCATION/id
        environment = FileCacheTrainingEnvironment(self.print_function,folder)
        task.info['run_at_dataset'] = dataset_version
        task.run_with_environment(self.routine.local_dataset_storage/dataset_version, environment)
        return id

    def get_result(self, id: str):
        return ResultPickleReader(_TRAINING_RESULTS_LOCATION/id, False)


def _create_id(prefix):
    dt = datetime.datetime.now()
    uid = str(uuid4()).replace('-', '')
    id = f'{prefix}_{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}_{uid}'
    return id
