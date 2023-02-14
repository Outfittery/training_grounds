from unittest import TestCase
from tg.common.datasets.featurization import *
from tg.common import Loc, MemoryFileSyncer
import os
import shutil
import pandas as pd
from datetime import datetime

root_path = Loc.temp_path / 'tests/updatable_dataset_processing_job_test_case'
src_path = root_path / 'dataset'
result_path = root_path / 'result'


def dt(i):
    return datetime(2020, 1, 1 + i)


def make_df(subpath, *args):
    buffer = []
    for i in range(0, len(args) - 1, 2):
        buffer.append(dict(index=args[i], value=args[i + 1]))
    fname = src_path / (subpath + '.parquet')
    os.makedirs(str(fname.parent), exist_ok=True)
    pd.DataFrame(buffer).set_index('index').to_parquet(fname)


class UpdatableDatasetProcessingJobTestCase(TestCase):
    def test_1(self):
        job = self._setup()
        job.run(current_time=dt(3), custom_revision_id='0')
        job.run(current_time=dt(5), custom_revision_id='1')
        job.run(current_time=dt(7), custom_revision_id='2')
        job.run(current_time=dt(9), custom_revision_id='3', force_full_update=True)
        job.dst_syncer.change_local_folder(result_path).download_folder('')

        self.ttest('0/a', 1, 110, 2, 120, 3, 130)
        self.ttest('0/b', 1, 1100, 2, 1200, 3, 1300)
        self.ttest('1/a', 1, 111, 2, 121)
        self.ttest('1/b', 1, 1110, 2, 1210)
        self.ttest('2/a', 2, 122, 3, 132)
        self.ttest('2/b', 2, 1220, 3, 1320)
        self.ttest('3/a', 2, 122, 3, 133)
        self.ttest('3/b', 2, 1220, 3, 1330)

        df = self.get_desc()
        self.assertEqual([True, False, True, True], list(df.is_major))

    def test_2(self):
        job = self._setup()
        job.run(current_time=dt(9), custom_revision_id='3')
        job.dst_syncer.change_local_folder(result_path).download_folder('')

        self.ttest('3/a', 2, 122, 3, 133)

        df = self.get_desc()
        self.assertListEqual([True], list(df.is_major))

    def test_3(self):
        job = self._setup()
        job.run(current_time=dt(5), custom_revision_id='1', custom_start_time=dt(3))
        job.dst_syncer.change_local_folder(result_path).download_folder('')

        self.ttest('1/a', 1, 111, 2, 121)

        df = self.get_desc()
        self.assertListEqual([False], list(df.is_major))

    def get_desc(self):
        return pd.read_parquet(result_path / UpdatableDataset.DescriptionHandler.get_description_filename())

    def ttest(self, path, *args):
        idx = []
        vls = []
        for i in range(0, len(args) - 1, 2):
            idx.append(args[i])
            vls.append(args[i + 1])
        df = Dataset(result_path / path, None).read().sort_index()

        try:
            self.assertListEqual(idx, list(df.index))
            self.assertListEqual(vls, list(df.value1))
        except:
            print(path)
            print(df)
            raise

    def _setup(self):
        make_df('0/a/1', 1, 10, 2, 20, 3, 30)
        make_df('0/b/1', 1, 100, 2, 200, 3, 300)
        make_df('1/a/1', 1, 11, 2, 21)
        make_df('1/b/1', 1, 110)
        make_df('1/b/2', 2, 210)
        make_df('2/a/1', 2, 22, 3, 32)
        make_df('2/b/1', 2, 220, 3, 320)
        make_df('3/a/1', 3, 33)
        make_df('3/b/1', 3, 330)

        records = [
            UpdatableDataset.DescriptionItem('0', dt(2), True, ''),
            UpdatableDataset.DescriptionItem('1', dt(4), False, ''),
            UpdatableDataset.DescriptionItem('2', dt(6), True, ''),
            UpdatableDataset.DescriptionItem('3', dt(8), False, ''),
        ]
        UpdatableDataset.DescriptionHandler.write_parquet(records, src_path / UpdatableDataset.DescriptionHandler.get_description_filename())

        src_syncer = MemoryFileSyncer(src_path)
        src_syncer.upload_folder('')

        shutil.rmtree(root_path, ignore_errors=True)
        os.makedirs(root_path)
        dst_syncer = MemoryFileSyncer()
        job = UpdatableDatasetScoringJob(
            'test',
            '',
            dst_syncer,
            [
                UpdatableDatasetScoringMethod('a', src_syncer, 'a', lambda df: df.assign(value1=df.value + 100)),
                UpdatableDatasetScoringMethod('b', src_syncer, 'b', lambda df: df.assign(value1=df.value + 1000)),
            ],
            root_path / 'job',
        )
        return job
