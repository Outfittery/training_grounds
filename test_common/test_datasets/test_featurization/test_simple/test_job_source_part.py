from typing import *
from unittest import TestCase
from tg.common.datasets.access.arch import AbstractCacheDataSource, Queryable, Query, MockDfDataSource, \
    CacheableDataSource
from tg.common.datasets.featurization import *
from tg.common import Loc, MemoryFileSyncer


class MyCacheDataSource(AbstractCacheDataSource):
    def __init__(self):
        self.available = False
        self.made = 0
        self.read = 0

    def cache_from(self, src: Queryable, cnt=None):
        self.made += 1
        self.available = True
        self.data = src.to_list()

    def get_data(self, **kwargs) -> Queryable:
        if not self.available:
            raise ValueError()
        self.read += 1
        return Query.en(self.data)

    def is_available(self):
        return self.available


data = Query.en(range(5)).select(lambda z: dict(a=z)).to_dataframe()


class MyFeaturizerSimple(DataframeFeaturizer):
    def __init__(self):
        super(MyFeaturizerSimple, self).__init__()

    def _featurize(self, item: Any) -> List[Any]:
        return [item]


class FeaturizationTestCase(TestCase):
    def test_cache_integration_1(self):
        cache = MyCacheDataSource()
        src = CacheableDataSource(
            MockDfDataSource(data),
            cache
        )
        mem = MemoryFileSyncer()
        job = FeaturizationJob(
            'test',
            'test',
            src,
            {
                'def': MyFeaturizerSimple()
            },
            mem,
            Loc.temp_path/'tests/featurization_job/source_part',
            None,
            None
        )
        self.assertRaises(ValueError, lambda: job.run(cache='use'))

        job.run(cache='default')
        self.assertEqual(True, cache.available)
        self.assertEqual(1, cache.made)
        self.assertEqual(1, cache.read)

        job.run(cache='default')
        self.assertEqual(1, cache.made)
        self.assertEqual(2, cache.read)

        job.run(cache='remake')
        self.assertEqual(2, cache.made)
        self.assertEqual(3, cache.read)

        job.run(cache='use')
        self.assertEqual(2, cache.made)
        self.assertEqual(4, cache.read)
