from unittest import TestCase
import pandas as pd
from yo_fluq_ds import *
from tg.common.datasets.featurization._common.index_filters import SimpleIndexFilter, PerformativeIndexFilter
from datetime import datetime

def make(f, t, *index):
    return Query.en(range(f,t)).select(lambda z: dict(a=z, b=z, c=z, z=z)).to_dataframe().set_index(list(index))

def make_large(i, N):
    line = list(range(i*N,(i+1)*N))
    return pd.DataFrame(dict(a=line,b=line,c=line,d=line))

TYPE = PerformativeIndexFilter

class IndexFilteringTestCase(TestCase):
    def test_simple(self):
        obj = TYPE()
        df, index = obj.filter(make(0,3, 'a'), None)
        self.assertListEqual([0,1,2], list(df.z))
        df, index = obj.filter(make(1,4,'a'), index)
        self.assertListEqual([3], list(df.z))

    def test_multi(self):
        obj = TYPE()
        df, index = obj.filter(make(0,3,'a','b'), None)
        self.assertListEqual([0, 1, 2], list(df.z))
        df, index = obj.filter(make(1,4,'a','b'), index)
        self.assertListEqual([3], list(df.z))
        self.assertEqual(4, len(index))
        df, index = obj.filter(make(1, 6, 'a', 'b'), index)
        self.assertListEqual([4,5], list(df.z))
        self.assertEqual(6, len(index))


    def dont_test_performance_one_column(self):
        N=100000
        index = None
        begin = datetime.now()
        obj = TYPE()
        for i in range(3):
            df = make_large(i,N).set_index('a')
            _, index = obj.filter(df, index)
        span = (datetime.now() - begin).total_seconds()
        self.assertLess(span,1)



    def dont_test_performance_2(self):
        N = 100000
        index = None
        begin = datetime.now()
        obj = TYPE()
        for i in range(3):
            df = make_large(i, N).set_index(['a','b'])
            _, index = obj.filter(df, index)
        span = (datetime.now() - begin).total_seconds()
        self.assertLess(span, 2)



