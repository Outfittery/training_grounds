from tg.common.ml.batched_training.context import *
from unittest import TestCase
import pandas as pd


class SmallComponentsTestCase(TestCase):
    def test_pandas_finalizer(self):
        original = pd.DataFrame(dict(sample_id=[1,2,3], v=[10,20,30])).set_index('sample_id')
        agg1 = pd.DataFrame(dict(sample_id=[1,2,3], a=[21,22,23])).set_index('sample_id')
        agg2 = pd.DataFrame(dict(sample_id=[1,2], b=[31,32])).set_index('sample_id')
        agg3 = pd.DataFrame(dict(sample_id=[1,3,4], b=[41,43,44])).set_index('sample_id')
        aggs = {'c1': agg1, 'c2': agg2, 'c3': agg3}
        fin = PandasAggregationFinalizer()
        fin.fit(original, {}, aggs)
        df = fin.finalize(original, {}, aggs)
        self.assertListEqual([21,22,23], list(df.c1_a))
        self.assertListEqual([31, 32, 0], list(df.c2_b.astype(int)))
        self.assertListEqual([41, 0, 43], list(df.c3_b.astype(int)))
        self.assertListEqual([1,1,1], list(df.c1_present_c1))
        self.assertListEqual([1, 1, 0], list(df.c2_present_c2))
        self.assertListEqual([1, 0, 1], list(df.c3_present_c3))


    def test_groupby_aggregator(self):
        df = pd.DataFrame(dict(
            sample_id = [1, 1, 1, 2, 2, 3],
            offset = [1, 2, 3, 2, 3, 1],
            value_1 = [0.1, 0.4, 0.1, 0.2, 0.2, 0.3],
            value_2 = [1, 2, 0, 1, 1, 0]
        )).set_index(['sample_id','offset'])
        agg = GroupByAggregator(['mean','max','min'])
        rdf = agg.aggregate_context(df)
        expected = {'value_1_mean': [0.19999999999999998, 0.2, 0.3], 'value_1_max': [0.4, 0.2, 0.3], 'value_1_min': [0.1, 0.2, 0.3], 'value_2_mean': [1, 1, 0], 'value_2_max': [2, 1, 0], 'value_2_min': [0, 1, 0]}
        rs = {c: list(rdf[c]) for c in rdf.columns}
        self.assertDictEqual(expected, rs)

    def test_pivot_aggregator(self):
        df = pd.DataFrame(dict(
            sample_id=[1, 1, 1, 2, 2, 3],
            offset=[1, 2, 3, 2, 3, 1],
            a = [1, 0, 0, 1, 0, 0],
            b = [0, 1, 1, 0, 1, 0],
            c = [0, 0, 0, 0, 0, 1]
        )).set_index(['sample_id','offset'])
        agg = PivotAggregator()
        rdf = agg.aggregate_context(df)
        expected = {'a_at_1': [1.0, 0.0, 0.0], 'a_at_2': [0.0, 1.0, 0.0], 'a_at_3': [0.0, 0.0, 0.0], 'b_at_1': [0.0, 0.0, 0.0], 'b_at_2': [1.0, 0.0, 0.0], 'b_at_3': [1.0, 1.0, 0.0], 'c_at_1': [0.0, 0.0, 1.0], 'c_at_2': [0.0, 0.0, 0.0], 'c_at_3': [0.0, 0.0, 0.0]}
        rs = {c: list(rdf[c]) for c in rdf.columns}
        self.assertDictEqual(expected, rs)


