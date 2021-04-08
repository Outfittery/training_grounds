import os
import boto3
import sagemaker

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


def _upload_container(task, project_name, id, image_tag, repository, region, push=True):
    packaging_task = pkg.PackagingTask(project_name, id, {'model': task})
    container_task = pkg.ContaineringTask(
        packaging_task,
        'train.py',
        _RUNNER_FILE_TEMPLATE,
        _DOCKERFILE_TEMPLATE,
        project_name,
        image_tag
    )
    pkg.make_container(container_task)
    if push:
        pkg.push_contaner_to_aws(project_name, image_tag, region, repository)


class SagemakerTrainingRoutine(TrainingRoutineBase):
    def __init__(self,
                 local_dataset_storage,
                 project_name: str,
                 aws_role: str,
                 ecr_repository: str,
                 region: str,
                 s3_bucket: str,
                 instance_type: str = 'ml.m4.xlarge',
                 use_spots=True,
                 ):
        self.project_name = project_name
        self.aws_role = aws_role
        self.use_spots = use_spots
        self.ecr_repository = ecr_repository
        self.region = region
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

        output_folder = Loc.temp_path / f'local_training/sagemaker/{id}'
        output_folder = os.path.abspath(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        image_tag = self.routine.project_name + '-' + id
        _upload_container(task, self.routine.project_name, id, image_tag, self.routine.ecr_repository,
                          self.routine.region, False)

        model = sagemaker.estimator.Estimator(
            self.routine.project_name + ":" + image_tag,
            self.routine.aws_role,
            train_instance_count=1,
            train_instance_type='local',
            output_path=f'file://{output_folder}'
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

    def _get_model(self, task, image_tag):
        sagemaker_name = task.info.get('name', self.routine.project_name).replace('_', '-')
        model = sagemaker.estimator.Estimator(
            self.routine.ecr_repository + ':' + image_tag,
            self.routine.aws_role,
            train_instance_count=1,
            train_instance_type=self.routine.instance_type,
            train_volume_size=5,
            output_path=f's3://{self.routine.s3_bucket}/sagemaker/{self.routine.project_name}/output/',
            sagemaker_session=self._get_session(),
            base_job_name=sagemaker_name,
            metric_definitions=get_sagemaker_metric_definitions(task)
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
        filename = _TRAINING_RESULTS_LOCATION / f'{id}.tar.gz'
        S3Handler.download_file(
            self.routine.s3_bucket,
            f'sagemaker/{self.routine.project_name}/output/{id}/output/model.tar.gz',
            filename
        )
        return ResultPickleReader.from_tar_gz_file(filename, id)

    def execute(self, task: AbstractTrainingTask, dataset_version: str, wait=True):
        id = _create_id(task.info.get('name', ''))
        task.info['run_at_dataset'] = dataset_version
        image_tag = self.routine.project_name + '-' + id
        _upload_container(task, self.routine.project_name, id, image_tag, self.routine.ecr_repository,
                          self.routine.region, True)
        model = self._get_model(task, image_tag)
        input_data = f's3://{self.routine.s3_bucket}/sagemaker/{self.routine.project_name}/datasets/{dataset_version}'
        model.fit(input_data)
        return model._current_job_name
