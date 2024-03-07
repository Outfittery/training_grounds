from unittest import TestCase
from tg.common.analysis import Aggregators
import pandas as pd
from yo_fluq_ds import Query
import numpy as np

df = (Query.combinatorics
      .grid(a=np.arange(0.1, 0.5, 0.2), b=np.arange(0, 0.5, 0.2))
      .select_many(lambda z: Query
                   .en(range(1000))
                   .select(lambda _: dict(p=np.random.random() < z.a + z.b, a=z.a, b=z.b))
                  )
      .to_dataframe()
     )
df['q'] = ~df.p

QFIELDS = ['q_lower', 'q_upper', 'q_value', 'q_error']
PFIELDS = ['p_lower', 'p_upper', 'p_value', 'p_error']


class AggregationTestCase(TestCase):
    def check_prop(self, obj, expected_index, expected_columns, force_columns=None):
        rdf = obj.feed(Aggregators.proportion_confint(force_columns))
        self.assertListEqual(expected_index, rdf.index.names)
        self.assertListEqual(expected_columns, list(rdf.columns))

    def test_grby_1(self):
        self.check_prop(df.drop('b', axis=1).groupby('a'), ['a'], PFIELDS + QFIELDS)

    def test_grby_2(self):
        self.check_prop(df.drop('b', axis=1).groupby(['a']), ['a'], PFIELDS + QFIELDS)

    def test_grby_3(self):
        self.check_prop(df.drop('b', axis=1).groupby(df.a), ['a'], PFIELDS + QFIELDS)

    def test_grby_4(self):
        self.check_prop(df.groupby(['a', 'b']), ['a', 'b'], PFIELDS + QFIELDS)

    def test_grby_5(self):
        self.check_prop(df.groupby(['b', 'a']), ['b', 'a'], PFIELDS + QFIELDS)

    def test_grby_and_access_1(self):
        self.check_prop(df.groupby('a')[['p', 'q']], ['a'], PFIELDS + QFIELDS)

    def test_grby_and_access_2(self):
        self.check_prop(df.groupby('a')[['p']], ['a'], PFIELDS)

    def test_grby_and_access_3(self):
        self.check_prop(df.groupby('a'), ['a'], PFIELDS, 'p')

    def test_grby_and_access_4(self):
        self.check_prop(df.groupby('a'), ['a'], PFIELDS, ['p'])

    def test_grby_series_1(self):
        self.check_prop(df.groupby('a').p, ['a'], PFIELDS)

    def test_grby_series_2(self):
        self.check_prop(df.groupby(['a', 'b']).p, ['a', 'b'], PFIELDS)

    def test_grby_series_3(self):
        self.assertRaises(ValueError, lambda: df.groupby('a').p.feed(Aggregators.proportion_confint(['p'])))

    def test_df_1(self):
        self.check_prop(df.drop(['a', 'b'], axis=1), [None], PFIELDS + QFIELDS)

    def test_df_2(self):
        self.check_prop(df[['p', 'q']], [None], PFIELDS + QFIELDS)

    def test_series(self):
        self.check_prop(df.p, [None], PFIELDS)
