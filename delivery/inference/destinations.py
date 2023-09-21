import os
from typing import *
from ..._common import Loc, S3Handler, DataBundle, Logger
import pandas as pd
import sqlalchemy as db
from yo_fluq_ds import Query, fluq
import uuid
from datetime import date
import os



class FakeBundleDestination:
    def upload_bundle(self, db: DataBundle):
        print(db)


class SqlBundleDestination:
    def __init__(self,
                 dataframe_name_in_bundle: str,
                 credentials_factory: Callable[[], Dict],
                 schema_name,
                 table_name,
                 columns_definition: Iterable[db.Column],
                 batch_size : int = 1000,
                 with_progress_bar: bool = False

    ):
        self.dataframe_name_in_bundle = dataframe_name_in_bundle
        self.credentials_factory = credentials_factory
        self.schema_name = schema_name
        self.table_name = table_name
        self.columns_definition = list(columns_definition)
        self.batch_size = batch_size
        self.with_progress_bar = with_progress_bar


    def _create_engine(self):
        c = self.credentials_factory()
        url = f"postgresql://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['dbname']}"
        engine = db.create_engine(url)
        return engine

    def _get_table(self, engine):
        meta = db.MetaData(schema=self.schema_name)
        if not db.inspect(engine).has_table(self.table_name, schema=self.schema_name):
                table = db.Table(
                    self.table_name,
                    meta,
                    db.Column('customer_id', db.Integer()),
                    db.Column('prediction_score', db.Float()),
                    db.Column('model_id', db.String(100)),
                    db.Column('updated_at', db.DateTime())
                )
                table.create(engine)
        else:
            table = db.Table(self.table_name, meta, autoload=True, autoload_with=engine)
        return table

    def _upload(self, engine, df):
        table = self._get_table(engine)
        en = Query.df(df).feed(fluq.partition_by_count(self.batch_size))
        if self.with_progress_bar:
            en = en.feed(fluq.with_progress_bar())

        with engine.connect() as conn:
            delete_statement = db.delete(table).compile(engine)
            conn.execute(delete_statement)
            for values in en:
                insert_statement = db.insert(table).values(values).compile(engine)
                conn.execute(insert_statement)

    def upload_bundle(self, db: DataBundle):
        Logger.info(f'Uploading frame {self.dataframe_name_in_bundle} to sql, schema {self.schema_name}, table {self.table_name}')
        engine = self._create_engine()
        self._upload(engine, db.data_frames[self.dataframe_name_in_bundle])
        Logger.info('Done')


class S3BundleDestination:
    def __init__(self, s3_bucket, s3_path, name = None):
        self.s3_bucket = s3_bucket
        self.s3_path = s3_path
        self.name = name
        if self.name is None:
            self.name = str(uuid.uuid4())


    def upload_bundle(self, db: DataBundle):
        Logger.info(f'Uploading bundle to {self.s3_bucket}//{self.s3_path}')
        os.makedirs(Loc.temp_path, exist_ok=True)
        db.save(Loc.temp_path/self.name)
        S3Handler.upload_folder(self.s3_bucket, self.s3_path, Loc.temp_path/self.name)


class S3TimestampedFileDestination:
    def __init__(self, s3_bucket, s3_path_template):
        self.s3_bucket = s3_bucket
        self.s3_path_template = s3_path_template

    def upload_bundle(self, db: DataBundle):
        path=Loc.temp_path/str(uuid.uuid4())
        os.makedirs(path.parent, exist_ok=True)
        db.save_as_zip(path)
        s3_path = self.s3_path_template.format(date.today().isoformat())
        S3Handler.upload_file(self.s3_bucket, s3_path, path)
