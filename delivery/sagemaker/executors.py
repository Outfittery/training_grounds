from .job import SagemakerJob
from ..delivery import Packaging, Containering
from ..._common import Loc
from .utils import open_sagemaker_result, download_and_open_sagemaker_result
from pathlib import Path
import datetime
import os
from uuid import uuid4
import sagemaker
import boto3


class SagemakerOptions:
    def __init__(self,
                 aws_role: str,
                 s3_bucket: str,
                 project_name: str,
                 local_datasets_folder: Path,
                 dataset_name: str
                 ):
        self.aws_role = aws_role
        self.s3_bucket = s3_bucket
        self.project_name = project_name
        self.local_datasets_folder = local_datasets_folder
        self.dataset_name = dataset_name
        self.instance_type = 'ml.m4.xlarge'
        self.use_spot_instances = False
        self.wait_until_completed = False

    def get_local_dataset_path(self):
        return self.local_datasets_folder / self.project_name / self.dataset_name



class SagemakerConfig:
    def __init__(self,
                 job: SagemakerJob,
                 packaging: Packaging,
                 containering: Containering,
                 sagemaker_settings: SagemakerOptions
                 ):
        self.job = job
        self.packaging = packaging
        self.containering = containering
        self.sagemaker_settings = sagemaker_settings




def _create_id(prefix):
    dt = datetime.datetime.now()
    uid = str(uuid4()).replace('-', '')
    id = f'{prefix}_{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}_{uid}'
    return id


class SagemakerAttachedExecutor:
    def __init__(self, config: SagemakerConfig):
        self.config = config

    def execute(self):
        input_path = self.config.sagemaker_settings.get_local_dataset_path()
        return self.config.job.task.run(input_path)


class SagemakerLocalExecutor:
    def __init__(self, config: SagemakerConfig):
        self.config = config


    def load_result(self, id):
        output_folder = Loc.temp_path / f'training_results/{id}'
        return open_sagemaker_result(output_folder/'model.tar.gz', id)




    def execute(self) -> str:
        id = _create_id(self.config.job.task.info.get('name', ''))
        self.config.job.task.info['run_at_dataset'] = self.config.sagemaker_settings.dataset_name

        output_folder = Loc.temp_path / f'training_results/{id}'
        output_folder = os.path.abspath(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        self.config.packaging.make_package()
        self.config.containering.make_container(self.config.packaging)

        model = sagemaker.estimator.Estimator(
            f"{self.config.containering.image_name}:{self.config.containering.image_tag}",
            self.config.sagemaker_settings.aws_role,
            instance_count=1,
            instance_type='local',
            output_path=f'file://{output_folder}',

        )
        model.fit(f'file://{self.config.sagemaker_settings.get_local_dataset_path().absolute()}')
        return id


class SagemakerRemoteExecutor:
    def __init__(self, config: SagemakerConfig):
        self.config = config

    @staticmethod
    def _get_sagemaker_metric_definitions(task):
        names = task.get_metric_names()
        metric_defs = [dict(Name=name, Regex=f"###METRIC###{name}:(.*?)###") for name in names]
        return metric_defs

    def _get_session(self):
        boto_session = boto3.Session(region_name='eu-west-1')
        session = sagemaker.Session(boto_session)
        return session

    def _get_model(self):
        sagemaker_name = self.config.containering.image_name.replace('_', '-')

        model = sagemaker.estimator.Estimator(
            self.config.containering.get_remote_name(),
            self.config.sagemaker_settings.aws_role,
            instance_count=1,
            instance_type=self.config.sagemaker_settings.instance_type,
            volume_size=5,
            output_path=f's3://{self.config.sagemaker_settings.s3_bucket}/sagemaker/{self.config.sagemaker_settings.project_name}/output/',
            sagemaker_session=self._get_session(),
            base_job_name=sagemaker_name,
            metric_definitions=self._get_sagemaker_metric_definitions(self.config.job.task),
        )
        if self.config.sagemaker_settings.use_spot_instances:
            model.use_spot_instances = self.config.sagemaker_settings.use_spot_instances
            model.max_wait = 86400
            model.max_run = 86400
        return model

    def get_result(self, id: str):
        return download_and_open_sagemaker_result(self.config.sagemaker_settings.s3_bucket, self.config.sagemaker_settings.project_name, id)

    def execute(self):
        self.config.job.task.info['run_at_dataset'] = self.config.sagemaker_settings.dataset_name
        self.config.packaging.make_package()
        self.config.containering.make_container(self.config.packaging)
        self.config.containering.push_container()
        model = self._get_model()
        input_data = f's3://{self.config.sagemaker_settings.s3_bucket}/sagemaker/{self.config.sagemaker_settings.project_name}/datasets/{self.config.sagemaker_settings.dataset_name}'
        model.fit(input_data, self.config.sagemaker_settings.wait_until_completed)
        return model._current_job_name

