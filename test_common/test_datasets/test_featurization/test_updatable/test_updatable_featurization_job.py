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

full_data_source = MockDfDataSource(Query.en(range(5)).select(lambda z: dict(a=z)).to_dataframe())
update_data_source = lambda _, __: MockDfDataSource(Query.en(range(3)).select(lambda z: dict(a=z)).to_dataframe())
featurizer_simple = DataframeFeaturizer(100, lambda z: z)
featurizer_batched = DataframeFeaturizer(2, lambda z: z)


def dt(i):
    return datetime(2020, 1, 1 + i)


class UpdatableFeaturizationJobTestCase(TestCase):
    def test_updatable_job(self):
        tmp_folder = Loc.temp_path / 'tests/updatable_featurization_job_test_case'
        shutil.rmtree(tmp_folder, ignore_errors=True)
        os.makedirs(tmp_folder)
        path1 = tmp_folder / 'local'
        mem = MemoryFileSyncer()

        job = UpdatableFeaturizationJob(
            'test',
            '1',
            full_data_source,
            update_data_source,
            dict(simple=featurizer_simple, batched=featurizer_batched),
            mem,
            path1,
            None,
            None
        )

        job.run(current_time=dt(2), custom_revision_id='2')
        job.run(current_time=dt(4), custom_revision_id='4')
        job.run(current_time=dt(6), force_major_update=True, custom_revision_id='6')

        rdf = pd.read_parquet(path1 / UpdatableDataset.DescriptionHandler.get_description_filename())
        self.assertEqual(['2', '4', '6'], list(rdf.name))
        self.assertEqual([True, False, True], list(rdf.is_major))

        s1 = pd.read_parquet(Query.folder(path1 / '2/simple').single())
        self.assertEqual(5, s1.shape[0])

        s2 = pd.read_parquet(Query.folder(path1 / '4/simple').single())
        self.assertEqual(3, s2.shape[0])

        s3 = pd.read_parquet(Query.folder(path1 / '6/simple').single())
        self.assertEqual(5, s3.shape[0])

        path2 = tmp_folder / 'dataset'
        ds = UpdatableDataset(
            path2,
            'simple',
            mem
        )

        rcs = ds.download(to_timestamp=dt(7))
        self.assertEqual(1, len(rcs))
        self.assertEqual('6', rcs[0].name)

        rcs = ds.download(to_timestamp=dt(5))
        self.assertEqual(2, len(rcs))
        self.assertListEqual(['4', '2'], [r.name for r in rcs])

        rcs = ds.download(to_timestamp=dt(3))
        self.assertEqual(1, len(rcs))
        self.assertEqual('2', rcs[0].name)

        rcs = ds.download(from_timestamp=dt(3), to_timestamp=dt(5))
        self.assertEqual(1, len(rcs))
        self.assertEqual('4', rcs[0].name)
