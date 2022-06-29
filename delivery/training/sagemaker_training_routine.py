from typing import *

import os
import boto3
import sagemaker
import shutil
import subprocess

from pathlib import Path

from .architecture import TrainingRoutineBase, TrainingExecutor, AttachedTrainingExecutor, AbstractTrainingTask, ResultPickleReader, _create_id, _TRAINING_RESULTS_LOCATION
from .. import packaging as pkg
from ..._common import Loc, S3Handler


_DOCKERFILE_TEMPLATE = '''FROM python:3.7

{install_libraries}

RUN pip install sagemaker-containers 

COPY . /opt/ml/code

WORKDIR /opt/ml/code

COPY {package_filename} package.tar.gz

RUN pip install package.tar.gz

WORKDIR /

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py
'''

_RUNNER_FILE_TEMPLATE = '''
import {module}.{tg_name}.common.delivery.training.sagemaker_execution as tg_sagemaker
from {module} import Entry

tg_sagemaker.execute(Entry)
'''


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
    folder = _TRAINING_RESULTS_LOCATION / job_id
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


def _upload_container(task, project_name, id, pusher: pkg.ContainerHandler, push=True):
    packaging_task = pkg.PackagingTask(project_name, id, {'model': task})
    container_task = pkg.ContaineringTask(
        packaging_task,
        'train.py',
        _RUNNER_FILE_TEMPLATE,
        _DOCKERFILE_TEMPLATE,
        pusher.get_image_name(),
        pusher.get_tag()
    )
    pkg.make_container(container_task)
    if push:
        pusher.push()


class SagemakerTrainingRoutine(TrainingRoutineBase):
    def __init__(self,
                 local_dataset_storage,
                 project_name: str,
                 handler_factory: pkg.ContainerHandler.Factory,
                 aws_role: str,
                 s3_bucket: str,
                 instance_type: str = 'ml.m4.xlarge',
                 use_spots=True,
                 ):
        self.project_name = project_name
        self.handler_factory = handler_factory
        self.aws_role = aws_role
        self.use_spots = use_spots
        self.s3_bucket = s3_bucket
        self.instance_type = instance_type
        super(SagemakerTrainingRoutine, self).__init__(local_dataset_storage)

        self.attached = AttachedTrainingExecutor(self)
        self.local = SagemakerLocalExecutor(self)
        self.remote = SagemakerRemoteExecutor(self)


class SagemakerLocalExecutor(TrainingExecutor):
    def __init__(self, routine: SagemakerTrainingRoutine):
        self.routine = routine

    def upload_dataset(self, dataset_version: str):
        pass

    def execute(self, task: AbstractTrainingTask, dataset_version: str, wait=True) -> str:
        id = _create_id(task.info.get('name', ''))
        task.info['run_at_dataset'] = dataset_version

        output_folder = Loc.temp_path / f'training_results/{id}'
        output_folder = os.path.abspath(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        handler = self.routine.handler_factory.create_handler(self.routine.project_name, id)
        _upload_container(task, self.routine.project_name, id, handler, False)

        model = sagemaker.estimator.Estimator(
            f"{handler.get_image_name()}:{handler.get_tag()}",
            self.routine.aws_role,
            train_instance_count=1,
            train_instance_type='local',
            output_path=f'file://{output_folder}',

        )
        model.fit(f'file://{self.routine.local_dataset_storage.absolute()}/{dataset_version}')
        return id

    def get_result(self, id: str):
        gzip_path = Loc.temp_path / f'local_training/sagemaker/{id}/model.tar.gz'
        return ResultPickleReader.from_tar_gz_file(gzip_path, id)


def get_sagemaker_metric_definitions(task: AbstractTrainingTask):
    """
    Creates metrics definition for sagemaker from :class:``TrainingTask``. The metrics are restored from :class:``MetricProvider`` instance within the class
    """
    names = task.get_metric_names()
    metric_defs = [dict(Name=name, Regex=f"###METRIC###{name}:(.*?)###") for name in names]
    return metric_defs


class SagemakerRemoteExecutor(TrainingExecutor):
    def __init__(self, routine: SagemakerTrainingRoutine):
        self.routine = routine

    def _get_session(self):
        boto_session = boto3.Session(region_name='eu-west-1')
        session = sagemaker.Session(boto_session)
        return session

    def _get_model(self, task, handler: pkg.ContainerHandler):
        sagemaker_name = task.info.get('name', self.routine.project_name).replace('_', '-')
        model = sagemaker.estimator.Estimator(
            handler.get_remote_name(),
            self.routine.aws_role,
            train_instance_count=1,
            train_instance_type=self.routine.instance_type,
            train_volume_size=5,
            output_path=f's3://{self.routine.s3_bucket}/sagemaker/{self.routine.project_name}/output/',
            sagemaker_session=self._get_session(),
            base_job_name=sagemaker_name,
            metric_definitions=get_sagemaker_metric_definitions(task),

        )
        if self.routine.use_spots:
            model.train_use_spot_instances = self.routine.use_spots,
            model.train_max_wait = 86400
        return model

    def upload_dataset(self, dataset_version: str):
        S3Handler.upload_folder(self.routine.s3_bucket,
                                f'sagemaker/{self.routine.project_name}/datasets/{dataset_version}',
                                self.routine.local_dataset_storage / dataset_version
                                )

    def download_dataset(self, dataset_version: str):
        S3Handler.download_folder(self.routine.s3_bucket,
                                  f'sagemaker/{self.routine.project_name}/datasets/{dataset_version}',
                                  self.routine.local_dataset_storage / dataset_version
                                  )

    def get_result(self, id: str):
        return download_and_open_sagemaker_result(self.routine.s3_bucket, self.routine.project_name, id)

    def execute(self, task: AbstractTrainingTask, dataset_version: str, wait=True):
        id = _create_id(task.info.get('name', ''))
        task.info['run_at_dataset'] = dataset_version
        handler = self.routine.handler_factory.create_handler(self.routine.project_name, id)
        image_tag = self.routine.project_name + '-' + id
        _upload_container(task, self.routine.project_name, id, handler, True)
        model = self._get_model(task, handler)
        input_data = f's3://{self.routine.s3_bucket}/sagemaker/{self.routine.project_name}/datasets/{dataset_version}'
        model.fit(input_data, wait)
        return model._current_job_name
