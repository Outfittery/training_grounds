from typing import *
from ..training_task import BatchedTrainingTask
from ....delivery.training import download_and_open_sagemaker_result

class AbstractBasisTaskSource:
    def load_task(self):
        raise NotImplementedError()

class SagemakerBasisTaskSource(AbstractBasisTaskSource):
    def __init__(self, bucket: str, project_name: str, job_id: str):
        self.bucket = bucket
        self.project_name = project_name
        self.job_id = job_id

    def load_task(self):
        rs = download_and_open_sagemaker_result(self.bucket, self.project_name, self.job_id, True)
        task = rs.unpickle('output/training_task.pkl')
        return task


