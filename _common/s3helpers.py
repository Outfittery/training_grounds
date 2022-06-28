from typing import *

import boto3
import shutil
import os

from pathlib import Path
from yo_fluq_ds import Query, fluq



class S3Handler:
    @staticmethod
    def download_file(bucket: str, s3_path: str, filename: Path):
        if s3_path.startswith('/'):
            s3_path = s3_path[1:]
        client = boto3.client('s3')
        os.makedirs(str(filename.parent),exist_ok=True)
        filename = filename.__str__()
        client.download_file(bucket, s3_path, filename)
        return filename


    @staticmethod
    def download_folder(bucket: str, s3_path: str, folder: Path, report=None):
        if os.path.exists(folder.__str__()):
            shutil.rmtree(folder.__str__())
        os.makedirs(folder.__str__())
        s3_resource = boto3.resource('s3')
        bucket_obj = s3_resource.Bucket(bucket)

        keys = [z.key for z in bucket_obj.objects.filter(Prefix=s3_path)]
        keys = Query.en(keys)
        if report == 'tqdm':
            keys = keys.feed(fluq.with_progress_bar())

        client = boto3.client('s3')

        for key in keys:
            proper_key = key[len(s3_path):]
            if proper_key.startswith('/'):
                proper_key=proper_key[1:]
            filename = folder.joinpath(proper_key)
            os.makedirs(str(filename.parent), exist_ok=True)
            filename = filename.__str__()
            client.download_file(bucket, key, filename)


    @staticmethod
    def upload_file(bucket_name: str, s3_path: str, filename: Union[Path,str]):
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID', None)
        aws_secret_access = os.environ.get('AWS_SECRET_ACCESS_KEY', None)
        if aws_access_key_id is not None and aws_secret_access is not None:
            kwargs = dict(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access)
        else:
            kwargs = {}

        client = boto3.client('s3', **kwargs)
        client.upload_file(str(filename), bucket_name, s3_path)


    @staticmethod
    def upload_folder(bucket_name: str, s3_path: str, folder: Path):
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID', None)
        aws_secret_access = os.environ.get('AWS_SECRET_ACCESS_KEY', None)
        if aws_access_key_id is not None and aws_secret_access is not None:
            kwargs = dict(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access)
        else:
            kwargs = {}

        s3 = boto3.resource('s3', **kwargs)
        bucket = s3.Bucket(bucket_name)
        bucket.objects.filter(Prefix=s3_path).delete()

        client = boto3.client('s3', **kwargs)
        for file_path in Query.folder(folder,'**/*'):
            if file_path.is_file():
                file_path_str = file_path.__str__()
                relative_path= file_path.relative_to(folder)
                joint_path = os.path.join(s3_path, relative_path)
                client.upload_file(file_path_str, bucket_name, joint_path)


    @staticmethod
    def remove_path(bucket_name: str, path: str, execute_action = False):
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID', None)
        aws_secret_access = os.environ.get('AWS_SECRET_ACCESS_KEY', None)
        if aws_access_key_id is not None and aws_secret_access is not None:
            kwargs = dict(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access)
        else:
            kwargs = {}

        s3 = boto3.resource('s3', **kwargs)
        bucket = s3.Bucket(bucket_name)

        if not execute_action:
            print(path)
        else:
            bucket.objects.filter(Prefix=path).delete()

    @staticmethod
    def get_latest_modified_file(bucket_name: str, path: str):
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID', None)
        aws_secret_access = os.environ.get('AWS_SECRET_ACCESS_KEY', None)
        if aws_access_key_id is not None and aws_secret_access is not None:
            kwargs = dict(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access)
        else:
            kwargs = {}

        s3 = boto3.client('s3', **kwargs)

        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=path,
        )
        if response['ResponseMetadata']['HTTPStatusCode'] == 200 and len(response['Contents']) > 0:
            max_modified = response['Contents'][0]['LastModified']
            path = response['Contents'][0]['Key']
            for object in response['Contents']:
                if object['LastModified'] > max_modified:
                    path = object['Key']
            return path
        return None
