from typing import *
import math

import pandas as pd
from sqlalchemy import create_engine, sql
from sqlalchemy.sql.selectable import Select
from hashlib import md5

from .._common import Logger
from .accessor import IAccessor
from yo_fluq_ds import Queryable, Query, fluq

class SQLAccessor(IAccessor[pd.DataFrame]):
    def __init__(
            self,
            query: Union[str, Select],
            dsn: str,
            chunk_size: Optional[int] = None,
            with_progress_bar: bool = False,
            chunk_concat_size: int = 100,
            with_log: bool = True,
    ):
        self.dsn = dsn
        self.query = query
        self._engine = create_engine(self.dsn)
        self.chunk_size = chunk_size
        self.with_progress_bar = with_progress_bar
        self.chunk_concat_size = chunk_concat_size
        self.log = with_log

    def log_message(self, msg: str):
        if self.log:
            Logger.info(msg)

    def _get_data_flow_iter(self):
        self.log_message(f'Executing query with chunks:\n{self.query}')
        with self._engine.connect().execution_options(stream_results=True) as conn:
            chunks = pd.read_sql_query(
                sql=self.query, con=conn, chunksize=self.chunk_size
            )
            if isinstance(chunks, pd.DataFrame):
                chunks = [chunks]
            for chunk in chunks:
                yield chunk

    def _create_count_query(self):
        if isinstance(self.query, str):
            return f'select count(*) from ({self.query}) as cnt_subquery'
        elif isinstance(self.query,Select):
            return sql.select([sql.func.count()]).select_from(self.query.subquery())
        else:
            raise TypeError(f'Expecting query to be str or Select, but was {type(self.query)}')


    def _get_data_flow_internal(self):
        if self.with_progress_bar:
            self.log_message('Counting rows')
            count_query = self._create_count_query()
            count = self._engine.execute(count_query).scalar()
            chunks_count = math.ceil(count/self.chunk_size)
            self.log_message(f'Rows {count}, chunk size {self.chunk_size}, chunks {chunks_count}')
            return Queryable(self._get_data_flow_iter(), length = chunks_count)
        else:
            self.log_message(f'Chunk size {self.chunk_size}')
            return Queryable(self._get_data_flow_iter())


    def get_data_flow(self) -> Queryable[pd.DataFrame]:
        if self.chunk_size is None:
            self.log_message(f"Executing query:\n{self.query}")
            df = pd.read_sql_query(self.query, self._engine)
            self.log_message(f"Done.")
            query = Query.en([df])
        else:
            query = self._get_data_flow_internal()
        if self.with_progress_bar:
            query = query.feed(fluq.with_progress_bar())
        return query

    def get_data(self) -> pd.DataFrame:
        chunks = self.get_data_flow()
        df = pd.DataFrame()
        chunks_to_concat = []
        for chunk in chunks:
            chunks_to_concat.append(chunk)
            if len(chunks_to_concat) >= self.chunk_concat_size:
                df = pd.concat([df] + chunks_to_concat)
                chunks_to_concat = []

        df = pd.concat([df] + chunks_to_concat)
        df = df.reset_index(drop=True)

        return df

    def get_name(self):
        if isinstance(self.query, str):
            s = self.query
        elif isinstance(self.query, Select):
            s = str(self.query.compile(compile_kwargs={'literal_binds': True}))
        else:
            raise TypeError(f'Expecting query to be str or Select, but was {type(self.query)}')
        return 'query_' + md5(s.encode('utf-8')).hexdigest()