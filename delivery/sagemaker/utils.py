from typing import *
from ..._common import Loc, S3Handler
from ..delivery import HackedUnpicker
from yo_fluq_ds import FileIO
from pathlib import Path
import os
import shutil
import subprocess


_TRAINING_RESULTS_LOCATION = Loc.temp_path / 'training_results'

class ResultPickleReader:
    def __init__(self, path: Path, hacked_unpickling=True):
        self.path = path
        self.hacked_unpickling = hacked_unpickling

    def unpickle(self, subpath, override_dst_module = None) -> Any:
        if self.hacked_unpickling:
            package = FileIO.read_jsonpickle(self.path.joinpath('package.json'))

            if 'tg_module_name' in package: #TODO: for compatibility only!
                from_module = package['tg_module_name']
                to_module = package['original_tg_module_name']
            else:
                from_module = package['tg_import_path']
                to_module = package['original_tg_import_path']
            if override_dst_module is not None:
                to_module = override_dst_module
            with open(str(self.path.joinpath(subpath)), 'rb') as file:
                unpickler = HackedUnpicker(file, from_module, to_module)
                return unpickler.load()
        else:
            return FileIO.read_pickle(self.path.joinpath(subpath))

    def get_path(self, subpath) -> Path:
        return self.path.joinpath(subpath)

    @staticmethod
    def from_tar_gz_file(path_to_tar_gz: Path, id: str, path_to_unpacked_folder: Optional[Path] = None):
        if path_to_unpacked_folder is None:
            path_to_unpacked_folder = Loc.temp_path.joinpath('training_jobs_results')
        folder = path_to_unpacked_folder / id
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        tar_call = ['tar', '-C', folder, '-xvf', path_to_tar_gz]
        subprocess.call(tar_call)
        return ResultPickleReader(Path(folder))


def open_sagemaker_result(filename, job_id):
    folder = _TRAINING_RESULTS_LOCATION / (job_id + '.unzipped')
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    tar_call = ['tar', '-C', folder, '-xvf', filename]
    subprocess.call(tar_call)
    return ResultPickleReader(Path(folder))


def download_and_open_sagemaker_result(bucket, project_name, job_id, dont_redownload=False):
    filename = _TRAINING_RESULTS_LOCATION / f'{job_id}.tar.gz'
    folder = _TRAINING_RESULTS_LOCATION / (job_id + '.unzipped')
    if filename.is_file() and folder.is_dir() and dont_redownload:
        return ResultPickleReader(Path(folder))
    else:
        path = f'sagemaker/{project_name}/output/{job_id}/output/model.tar.gz'
        S3Handler.download_file(
            bucket,
            path,
            filename
        )
        return open_sagemaker_result(filename, job_id)

