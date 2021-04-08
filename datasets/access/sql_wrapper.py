from typing import *

import datetime
import logging

from datetime import timedelta
from yo_fluq_ds import Query, fluq

from .arch import DataSource



class SqlShardedDataSource(DataSource):
    """
    Abstract class that represents a sharded query.
    Instead of the query, it accepts the template. This template is to be converted to the sequence of queries
    by descendands of this class. For each query, a :class:``DataSource`` will be created with ``downloader_factory``.
    This DataSources will then be executed, their results concatenated.

    The purpose of this class is to make SQL queries more responsive. Each shard is executed in a shorter time,
    thus it is possible to estimate the total time for the query.

    It is unknown if sharding reduces the overall time of the query.

    """

    def __init__(self, query_template: str, downloader_factory: Callable[[str], DataSource]):
        self.query_template = query_template
        self.downloader_factory = downloader_factory
        self.with_progress_bar = False

    def get_splits(self, query_template) -> List[str]:
        """
        Inheritants of this class must return the list of generated queries for shards from this method
        """
        raise NotImplementedError()

    def get_data(self):
        splits = list(self.get_splits(self.query_template))
        query = Query.en(splits)
        if self.with_progress_bar:
            query = query.feed(fluq.with_progress_bar())
        query = query.select_many(lambda z: self.downloader_factory(z).get_data())
        return query


class IntFieldShardedJob(SqlShardedDataSource):
    """
    This class represents int-based sharding. The entities are typically separated into shards by
    [some_id_field] % {shard_count} = {shard} condition
    """

    def __init__(self, query_template, downloaded_factory, shard_count, custom_shards: Optional[List[int]] = None):
        super(IntFieldShardedJob, self).__init__(query_template, downloaded_factory)
        self.shard_count = shard_count
        self.custom_shards = custom_shards

    def get_splits(self, query_template):
        if self.custom_shards is not None:
            shards = self.custom_shards
        else:
            shards = list(range(self.shard_count))
        return (Query
                .en(shards)
                .select(lambda z: dict(shard=z, shard_count=self.shard_count))
                .select(lambda z: query_template.format(**z))
                .to_list()
                )


logger = logging.getLogger(__name__)


class UpdateDataSource:
    def __init__(self,
                 id_retrieve_sql_template: str,
                 download_sql_template: str,
                 partition_size: int,
                 source_factory: Callable[[str], DataSource],
                 start_date: datetime.datetime,
                 end_date: datetime.datetime
                 ):
        self.id_retrieve_sql_template = id_retrieve_sql_template
        self.download_sql_template = download_sql_template
        self.partition_size = partition_size
        self.source_factory = source_factory
        self.start_date = start_date
        self.end_date = end_date

    def _get_data_iter(self, start_date: datetime.datetime, end_date: datetime.datetime):
        start_date_str = str(start_date)
        end_date_str = str(end_date)
        logger.info(f"Retrieving updated ids from {start_date_str} to {end_date_str}")
        sql = self.id_retrieve_sql_template.format(start_date=start_date_str, end_date=end_date_str)
        id_src = self.source_factory(sql)
        ids = id_src.get_data().select(lambda z: z['id']).select(str).to_list()
        partitions = Query.en(ids).feed(fluq.partition_by_count(self.partition_size)).to_list()
        logger.info(f'Retrieving {len(ids)} records, {len(partitions)} partitions')
        for index, partition in enumerate(partitions):
            id_list = ','.join(partition)
            sql = self.download_sql_template.format(id_list=id_list)
            src = self.source_factory(sql)
            for item in src.get_data():
                yield item
            logger.info(f"Partition {index} is processed")

    def get_data(self):
        return Query.en(self._get_data_iter(self.start_date, self.end_date))


class DayShardedSource(SqlShardedDataSource):
    """
    Sharder for data source that queries data for each particular day. Important for Impala data source,
    because data in out data warehouse is sharded on this basis
    """

    def __init__(self, sql_template, date_from: datetime, date_to: datetime, src_generator):
        self.date_from = date_from
        self.date_to = date_to
        super(DayShardedSource, self).__init__(sql_template, src_generator)

    def get_splits(self, query_template, **kwargs):
        splits = (Query
                  .loop(self.date_from, timedelta(days=1), self.date_to)
                  .select(lambda z: dict(day=z.day, month=z.month, year=z.year))
                  .select(lambda z: query_template.format(**z))
                  .to_list()
                  )
        return splits
