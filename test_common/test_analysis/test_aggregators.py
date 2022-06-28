from unittest import TestCase
from tg.common.analysis import Aggregators
import pandas as pd
from yo_fluq_ds import Query
import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint


df = (Query.combinatorics
      .grid(a=np.arange(0.1,0.5,0.2), b=np.arange(0,0.5,0.2))
      .select_many(lambda z: Query
                   .en(range(1000))
                   .select(lambda _: dict(p=np.random.random()<z.a+z.b, a=z.a, b=z.b))
                  )
      .to_dataframe()
     )
df['q'] = ~df.p

class AggregatorTestCase(TestCase):
    def test_proportion_1(self):
        rdf = df.groupby(['a','b']).feed(Aggregators.proportion_confint()).reset_index()
        rdf['delta'] = rdf.p_value-rdf.a - rdf.b
        self.assertTrue( (rdf.delta.abs()<0.1).all())

    def test_pandas(self):
        rdf = df.groupby('a').feed(Aggregators.pandas(p='mean', q=['mean','std']))
        self.assertListEqual(['p_mean','q_mean','q_std'], list(rdf.columns))

    def test_percentile(self):
        df = Query.en([0,10]).select_many(lambda z: Query.en(norm.rvs(size=1000,loc=z, scale=0.1)).select(lambda y: dict(a=z, p=y))).to_dataframe()
        rdf = df.groupby('a').feed(Aggregators.percentile_confint())
        self.assertTrue( ((rdf.p_value-rdf.index).abs()<0.1).all())


    def test_percentile_multilevel(self):
        df = Query.en([0, 10]).select_many(lambda z: Query.en(norm.rvs(size=1000, loc=z, scale=0.1)).select(
            lambda y: dict(a=z, p=y))).to_dataframe()
        rdf = df.groupby('a').feed(Aggregators.percentile_confint(multilevel_column=True))
        self.assertIsInstance(rdf.columns, pd.MultiIndex)

    def test_proportion_confint(self):
        success = 23
        failures = 124
        df = pd.DataFrame(dict(v=[True]*success+[False]*failures))
        rs = df.feed(Aggregators.proportion_confint(pValue=0.9)).transpose()
        true = proportion_confint(success, success+failures, 0.1)
        self.assertEqual(rs.loc['v_lower'].iloc[0], true[0])
        self.assertEqual(rs.loc['v_upper'].iloc[0], true[1])
        self.assertEqual(rs.loc['v_value'].iloc[0], (true[0] + true[1]) / 2)
        self.assertEqual(rs.loc['v_error'].iloc[0], (true[1] - true[0]) / 2)

    def test_combination(self):
        rdf = df.groupby(['a','b']).feed(
            Aggregators.proportion_confint() +
            Aggregators.pandas(p=['mean','std'])
        )
        self.assertListEqual(
            ['p_lower', 'p_upper', 'p_value', 'p_error', 'q_lower', 'q_upper', 'q_value', 'q_error', 'p_mean', 'p_std'],
            list(rdf.columns)
        )



