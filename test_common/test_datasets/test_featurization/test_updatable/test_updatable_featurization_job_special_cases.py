from unittest import TestCase
from tg.common.datasets.featurization import UpdatableFeaturizationJob
from tg.common.datasets.featurization import DataframeFeaturizer
from tg.common import Loc, MemoryFileSyncer
from tg.common.datasets.access import MockDfDataSource
from tg.common.datasets.featurization import UpdatableDataset
import os
import shutil
from yo_fluq_ds import Query
import pandas as pd
from pprint import pprint
from datetime import datetime


class UpdatableFeaturizationJobTestCase(TestCase):
    def test_updatable_job(self):
        empty_df = pd.DataFrame([])
        full_data_source = MockDfDataSource(empty_df)
        update_data_source = lambda _, __: MockDfDataSource(empty_df)
        featurizer_simple = DataframeFeaturizer(100, lambda z: z)
        featurizer_batched = DataframeFeaturizer(2, lambda z: z)
        mem = MemoryFileSyncer()

        job = UpdatableFeaturizationJob(
            'test',
            '',
            full_data_source,
            update_data_source,
            {
                'simple': featurizer_simple,
                'batched': featurizer_batched
            },
            mem,
            Loc.temp_path/'tests/updatable_featurization_job_empty'
        )
        job.run()
        self.assertEqual(0, len(mem.cache))



