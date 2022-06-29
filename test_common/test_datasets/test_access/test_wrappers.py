from unittest import TestCase
from tg.common.datasets.access import *
from yo_fluq_ds import Query


class MockSource(DataSource):
    state = 0

    def __init__(self, query):
        if query != f'10,{MockSource.state}':
            raise ValueError(query)
        MockSource.state += 1

    def get_data(self, **kwargs):
        return Query.en([1])


class MockUpdateSource(DataSource):
    state = 0

    def __init__(self, query):
        if (MockUpdateSource.state == 0 and query != 'id start end' or
                MockUpdateSource.state == 1 and query != '2,1,0' or
                MockUpdateSource.state == 2 and query != '3,4,5'):
            raise ValueError(f"{MockUpdateSource.state}, {query}")
        MockUpdateSource.state += 1

    def get_data(self):
        if MockUpdateSource.state == 1:
            return Query.en([2, 1, 0, 3, 4, 5]).select(lambda z: dict(id=z))
        else:
            return Query.en([])


class SqlWrappersTestCase(TestCase):
    def test_int_split(self):
        MockSource.state = 0
        src = IntFieldShardedJob('{shard_count},{shard}', MockSource, 10)
        data = src.get_data().to_list()
        self.assertEqual([1] * 10, data)

    def test_int_split_custom_shards(self):
        MockSource.state = 3
        src = IntFieldShardedJob('{shard_count},{shard}', MockSource, 10, [3, 4, 5])
        data = src.get_data().to_list()
        self.assertEqual([1] * 3, data)

    def test_update_wrapper(self):
        src = UpdateDataSource(
            'id {start_date} {end_date}',
            '{id_list}',
            3,
            MockUpdateSource,
            'start',
            'end'
        )
        data = src.get_data().to_list()
        self.assertListEqual([], data)
