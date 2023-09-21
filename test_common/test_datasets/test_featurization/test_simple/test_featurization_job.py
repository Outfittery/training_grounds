from typing import *
from tg.common.datasets.featurization import *
from tg.common.datasets.access.arch import MockDfDataSource, CacheableDataSource
from tg.common import MemoryFileSyncer, Loc
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
        mem = MemoryFileSyncer()
        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(data),
            {
                'def': MyFeaturizerSimple()
            },
            mem,
            Loc.temp_path / 'tests/featurization_job/simple',
            None,
            None
        )
        job.run()
        files = list(mem.cache)
        self.assertEqual(1, len(files))
        self.assertTrue(files[0].startswith('def'+Loc.file_slash))
        self.assertListEqual([0, 1, 2, 3, 4], list(mem.get_parquet(0).a))
        self.assertEqual(5, job.records_processed_)

    def test_empty_source(self):
        mem = MemoryFileSyncer()
        empty_df = pd.DataFrame([])
        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(empty_df),
            {
                'def': MyFeaturizerSimple()
            },
            mem,
            Loc.temp_path / 'tests/featurization_job/simple',
            None,
            None
        )
        job.run()
        files = list(mem.cache)
        self.assertEqual(0, len(files))
        self.assertEqual(0, job.records_processed_)

    def test_batched(self):
        mem = MemoryFileSyncer()
        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(data),
            {
                'def': MyFeaturizerBatch()
            },
            mem,
            Loc.temp_path / 'tests/featurization_job/batched',
            None,
            None
        )
        job.run()
        files = list(mem.cache)
        self.assertEqual(3, len(files))
        inds = [tuple(mem.get_parquet(i).a) for i in range(3)]
        self.assertIn((0, 1), inds)
        self.assertIn((2, 3), inds)
        self.assertIn((4,), inds)

    def test_two(self):
        mem = MemoryFileSyncer()
        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(data),
            {
                'simple': MyFeaturizerSimple(),
                'batched': MyFeaturizerBatch()
            },
            mem,
            Loc.temp_path / 'tests/featurization_job/two',
            None,
            None
        )
        job.run()
        stats = Query.en(mem.cache).group_by(lambda z: z.split(Loc.file_slash)[0]).to_dictionary(lambda z: z.key, lambda z: len(z.value))
        self.assertDictEqual(
            {'batched': 3, 'simple': 1},
            stats
        )

    def test_failing(self):
        mem = MemoryFileSyncer()
        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(data),
            {
                'fail': MyFeaturizerFailing()
            },
            mem,
            Loc.temp_path / 'tests/featurization_job/failing',
            None,
            None
        )
        self.assertRaises(ValueError, lambda: job.run())

    def test_postprocess(self):
        mem = MemoryFileSyncer()
        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(data),
            {
                'postprocess': MyFeaturizerPostprocess()
            },
            mem,
            Loc.temp_path / 'tests/featurization_job/postprocess',
            None,
            None
        )
        job.run()
        files = list(mem.cache)
        self.assertEqual(1, len(files))
        self.assertEqual('a', mem.get_parquet(0).index.name)
