from typing import *
from unittest import TestCase
from tg.common.datasets.featurization import *
from tg.common.datasets.access.arch import MockDfDataSource, CacheableDataSource
from tg.common import Loc, MemoryFileSyncer
from yo_fluq_ds import Query
import shutil
import os

data = Query.en(range(5)).select(lambda z: dict(a=z)).to_dataframe()


class MyFeaturizerSimple(DataframeFeaturizer):
    def __init__(self):
        super(MyFeaturizerSimple, self).__init__()

    def _featurize(self, item: Any) -> List[Any]:
        return [item]


class MyFeaturizerBatch(DataframeFeaturizer):
    def __init__(self):
        super(MyFeaturizerBatch, self).__init__(2)

    def _featurize(self, item: Any) -> List[Any]:
        return [item]


def get_files(location):
    return Query.folder(location, '**/*').where(lambda z: z.is_file()).select(lambda z: z.relative_to(location)).order_by(lambda z: z).to_list()


class FeaturizationJobTestCase(TestCase):
    def test_two(self):
        path1 = Loc.temp_path / 'tests/featurization_job_mem_test_case/local'
        path2 = Loc.temp_path / 'tests/featurization_job_mem_test_case/control'
        path3 = Loc.temp_path / 'tests/featurization_job_mem_test_case/dataset'
        shutil.rmtree(path1.parent, ignore_errors=True)
        os.makedirs(path1.parent)

        cache = dict()
        mem = MemoryFileSyncer(cache=cache)

        job = FeaturizationJob(
            'test',
            'test',
            MockDfDataSource(data),
            {
                'simple': MyFeaturizerSimple(),
                'batched': MyFeaturizerBatch()
            },
            mem,
            path1,
            None,
            None
        )
        job.run()

        MemoryFileSyncer(path2, '', mem.cache).download_folder('')

        self.assertListEqual(get_files(path1), get_files(path2))

        ds = Dataset(path3, MemoryFileSyncer(None, 'batched', mem.cache))
        ds.download()
        self.assertListEqual(get_files(path1 / 'batched'), get_files(path3))
