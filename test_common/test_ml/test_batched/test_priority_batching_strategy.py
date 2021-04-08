from tg.common.ml.batched_training import *
from unittest import TestCase
import pandas as pd
from yo_fluq_ds import *
import numpy as np

class PRBSTestCase(TestCase):
    def test_sampling(self):
        lst = list(range(4))
        index = pd.Series([100+c for c in lst])
        prios = pd.DataFrame(dict(prio=lst), index=index)
        strategy = PriorityRandomBatcherStrategy('prio',1)
        N=10000
        result = strategy.get_batch(N,prios,0)
        self.assertIsInstance(result,pd.Int64Index)
        r = result.to_frame().groupby(0).size()/N
        s = 1+2+3
        self.assertLess(abs(1/s-r[101]), 0.1)
        self.assertLess(abs(2 / s - r[102]), 0.1)
        self.assertLess(abs(3 / s - r[103]), 0.1)

    def make_evening_result(self, magnitude):
        df = pd.DataFrame(dict(key=range(2,32)))
        df['value'] = (np.log(df.key)/np.log(2)).astype(int)
        df = df.set_index('key')
        df = df.sample(frac=1)
        prios = PriorityRandomBatcherStrategy.make_priorities_for_even_representation(df,'value', magnitude)
        df['prio'] = prios
        strategy = PriorityRandomBatcherStrategy('prio',1)
        N=10000
        result=strategy.get_batch(N,df,0)
        x = df.loc[result]
        x = x.groupby('value').feed(fluq.fractions())
        return x

    def test_evening(self):
        x = self.make_evening_result(1)
        x = np.abs((x-0.25)).max()
        self.assertLess(x,0.05)



    def test_evening_magnitude_0_5(self):
        x = self.make_evening_result(0.5)
        self.assertLess(0.06,x.loc[1])
        self.assertLess(x.loc[1],x.loc[4])

    def test_evening_magnitude_2(self):
        x = self.make_evening_result(5)
        self.assertLess(0.06,x.loc[1])
        self.assertLess(x.loc[3],x.loc[1])








