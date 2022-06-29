from unittest import TestCase
from tg.common.datasets.access import *
from yo_fluq_ds import Query


class MockDataSource(DataSource):
    def __init__(self):
        self.access_count = 0

    def get_data(self):
        self.access_count += 1
        return Query.en([1, 2, 3])


class MockCache(AbstractCacheDataSource):
    def __init__(self):
        super(MockCache, self).__init__()
        self.reads_count = 0
        self.writes_count = 0
        self.cache = None

    def cache_from(self, src, cnt=None):
        self.cache = src.to_list()
        self.writes_count += 1

    def is_available(self) -> bool:
        return self.cache is not None

    def get_data(self):
        self.reads_count += 1
        return Query.en(self.cache)


def create_source(default=None):
    return CacheableDataSource(
        MockDataSource(),
        MockCache(),
        default
    )


class CacheTestCase(TestCase):
    def check(self, src, access_count, writes_count, reads_count):
        self.assertEqual(
            (access_count, writes_count, reads_count),
            (src._inner_datasource.access_count, src._file_data_source.writes_count, src._file_data_source.reads_count)
        )

    def no_cache_scenarios(self, src_factory, getter):
        src = src_factory()
        self.check(src, 0, 0, 0)
        getter(src).get_data().to_list()
        self.check(src, 1, 0, 0)
        getter(src).get_data().to_list()
        self.check(src, 2, 0, 0)

    def use_cache_scenario_0(self, src_factory, getter):
        src = src_factory()
        self.check(src, 0, 0, 0)
        self.assertRaises(ValueError, lambda: getter(src).get_data().to_list())

    def use_cache_scenario_1(self, src_factory, getter):
        src = src_factory()
        self.check(src, 0, 0, 0)
        src.safe_cache(CacheMode.Default).get_data().to_list()
        self.check(src, 1, 1, 1)
        getter(src).get_data().to_list()
        self.check(src, 1, 1, 2)

    def use_cache_scenario_2(self, src_factory, getter):
        src = src_factory()
        self.check(src, 0, 0, 0)
        src.make_cache()
        self.check(src, 1, 1, 0)
        getter(src).get_data().to_list()
        self.check(src, 1, 1, 1)

    def use_cache_scenarios(self, src_factory, getter):
        self.use_cache_scenario_0(src_factory, getter)
        self.use_cache_scenario_1(src_factory, getter)
        self.use_cache_scenario_2(src_factory, getter)

    def remake_scenarios(self, src_factory, getter):
        src = src_factory()
        self.check(src, 0, 0, 0)
        getter(src).get_data().to_list()
        self.check(src, 1, 1, 1)
        getter(src).get_data().to_list()
        self.check(src, 2, 2, 2)

    def default_scenarios(self, src_factory, getter):
        src = src_factory()
        self.check(src, 0, 0, 0)
        getter(src).get_data().to_list()
        self.check(src, 1, 1, 1)
        getter(src).get_data().to_list()
        self.check(src, 1, 1, 2)

    def test_no_string(self):
        self.no_cache_scenarios(
            lambda: create_source(),
            lambda s: s.safe_cache('no')
        )

    def test_no_enum(self):
        self.no_cache_scenarios(
            lambda: create_source(),
            lambda s: s.safe_cache(CacheMode.No)
        )

    def test_no_default(self):
        self.no_cache_scenarios(
            lambda: create_source(CacheMode.No),
            lambda s: s
        )

    def test_use_string(self):
        self.use_cache_scenarios(
            lambda: create_source(),
            lambda s: s.safe_cache('use')
        )

    def test_use_enum(self):
        self.use_cache_scenarios(
            lambda: create_source(),
            lambda s: s.safe_cache(CacheMode.Use)
        )

    def test_use_default(self):
        self.use_cache_scenarios(
            lambda: create_source(CacheMode.Use),
            lambda s: s
        )

    def test_remake_string(self):
        self.remake_scenarios(
            lambda: create_source(),
            lambda s: s.safe_cache('remake')
        )

    def test_remake_enum(self):
        self.remake_scenarios(
            lambda: create_source(),
            lambda s: s.safe_cache(CacheMode.Remake)
        )

    def test_remake_default(self):
        self.remake_scenarios(
            lambda: create_source(CacheMode.Remake),
            lambda s: s
        )

    def test_default_string(self):
        self.default_scenarios(
            lambda: create_source(),
            lambda s: s.safe_cache('default')
        )

    def test_default_enum(self):
        self.default_scenarios(
            lambda: create_source(),
            lambda s: s.safe_cache(CacheMode.Default)
        )

    def test_default_default(self):
        self.default_scenarios(
            lambda: create_source(CacheMode.Default),
            lambda s: s
        )

    def test_default_by_default(self):
        self.default_scenarios(
            lambda: create_source(),
            lambda s: s
        )
