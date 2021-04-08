from typing import *
from tg.common.datasets.featurization import *
from tg.common.datasets.access.arch import MockDfDataSource, CacheableDataSource
from unittest import TestCase
from yo_fluq_ds import Query, Queryable
import pandas as pd


class MyFeaturizerBatch(DataframeFeaturizer):
    def __init__(self):
        super(MyFeaturizerBatch, self).__init__(2)

    def _featurize(self, item: Any) -> List[Any]:
        return [item]


class MyFeaturizerSimple(DataframeFeaturizer):
    def __init__(self):
        super(MyFeaturizerSimple, self).__init__()

    def _featurize(self, item: Any) -> List[Any]:
        return [item]


class MyFeaturizerPostprocess(DataframeFeaturizer):
    def __init__(self):
        super(MyFeaturizerPostprocess, self).__init__()

    def _featurize(self, item: Any) -> List[Any]:
        return [item]

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.set_index('a')


class MyFeaturizerFailing(DataframeFeaturizer):
    def __init__(self):
        super(MyFeaturizerFailing, self).__init__()

    def _featurize(self, item: Any) -> List[Any]:
        return [item]

    def _validate(self):
        raise ValueError()


data = Query.en(range(5)).select(lambda z: dict(a=z)).to_dataframe()


class BatchJobTestCase(TestCase):
    def test_simple(self):
        mem = InMemoryJobDestination()
        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(data),
            {
                'def': MyFeaturizerSimple()
            },
            mem,
            None,
            None
        )
        job.run()
        self.assertSetEqual(set(mem.buffer), {'def'})
        self.assertEqual(1, len(mem.buffer['def']))
        self.assertListEqual([0, 1, 2, 3, 4], list(mem.buffer['def'][0].a))

    def test_batched(self):
        mem = InMemoryJobDestination()
        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(data),
            {
                'def': MyFeaturizerBatch()
            },
            mem,
            None,
            None
        )
        job.run()
        self.assertSetEqual(set(mem.buffer), {'def'})
        self.assertEqual(3, len(mem.buffer['def']))
        self.assertListEqual([0, 1], list(mem.buffer['def'][0].a))
        self.assertListEqual([2, 3], list(mem.buffer['def'][1].a))
        self.assertListEqual([4], list(mem.buffer['def'][2].a))

    def test_two(self):
        mem = InMemoryJobDestination()
        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(data),
            {
                'simple': MyFeaturizerSimple(),
                'batched': MyFeaturizerBatch()
            },
            mem,
            None,
            None
        )
        job.run()
        self.assertSetEqual(set(mem.buffer), {'simple', 'batched'})
        self.assertEqual(1, len(mem.buffer['simple']))
        self.assertEqual(3, len(mem.buffer['batched']))

    def test_failing(self):
        mem = InMemoryJobDestination()
        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(data),
            {
                'fail': MyFeaturizerFailing()
            },
            mem,
            None,
            None
        )
        self.assertRaises(ValueError, lambda: job.run())

    def test_postprocess(self):
        mem = InMemoryJobDestination()
        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(data),
            {
                'postprocess': MyFeaturizerPostprocess()
            },
            mem,
            None,
            None
        )
        job.run()
        self.assertSetEqual(set(mem.buffer), {'postprocess'})
        self.assertEqual(1, len(mem.buffer['postprocess']))
        self.assertEqual('a', mem.buffer['postprocess'][0].index.name)
