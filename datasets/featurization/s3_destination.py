import os
import boto3

from yo_fluq_ds import Query

from .featurization_job import LocalFileJobDestination
from ..._common import Loc



class S3FeaturizationJobDestination(LocalFileJobDestination):
    def __init__(self,
                 s3_bucket: str,
                 s3_path: str
                 ):
        self.s3_bucket = s3_bucket
        self.s3_path = s3_path
        loc = Loc.temp_path.joinpath(f'featurization_job_s3_destination/{s3_path}')
        super(S3FeaturizationJobDestination, self).__init__(loc)

    def send(self):
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID',None)
        aws_secret_access=os.environ.get('AWS_SECRET_ACCESS_KEY',None)
        if aws_access_key_id is not None and aws_secret_access is not None:
            kwargs=dict(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access)
        else:
            kwargs = {}


        s3 = boto3.resource('s3',**kwargs)
        bucket = s3.Bucket(self.s3_bucket)

        for name in self.names:
            s3_full_path = os.path.join(self.s3_path,name)
            bucket.objects.filter(Prefix=s3_full_path).delete()

        client = boto3.client('s3', **kwargs)
        for name in self.names:
            for file_path in Query.folder(self.location.joinpath(name)):
                client.upload_file(file_path.__str__(), self.s3_bucket, os.path.join(self.s3_path,name, file_path.name))

