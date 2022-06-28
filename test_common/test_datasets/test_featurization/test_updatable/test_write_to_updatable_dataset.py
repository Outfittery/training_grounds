from tg.common.datasets.featurization import UpdatableDataset
from tg.common import MemoryFileSyncer, Loc
from unittest import TestCase
import pandas as pd
from datetime import datetime

def get_df(v):
    return pd.DataFrame([dict(v=1)])

class UpdatableDatasetWriteTestCase(TestCase):
    def test_updatable_dataset_writing(self):
        N = 2
        data = {}
        data['a'] = [get_df(i) for i in range(N)]
        data['b'] = [get_df(i*10) for i in range(N)]
        records = [UpdatableDataset.DescriptionItem(str(i),datetime(2021,10,1+2*i), i%2==0, '1') for i in range(2)]

        syncer = MemoryFileSyncer()
        for i in range(N):
            UpdatableDataset.write_to_updatable_dataset(syncer,records[i],dict(a=data['a'][i],b=data['b'][i]))

        for i in range(N):
            for key in ['a','b']:
                ds = UpdatableDataset(Loc.temp_path/'tests/updatable_dataset_writer/', key, syncer)
                df = ds.read(to_timestamp=datetime(2021,10,1+2*i+1), cache_mode='remake')
                self.assertListEqual(list(data[key][i].v), list(df.v))



