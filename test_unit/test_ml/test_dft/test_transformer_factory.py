from unittest import TestCase
import pandas as pd
from tg.common.ml.dft import DataFrameTransformerFactory, ContinousTransformer, DataFrameColumnsTransformer
import numpy as np
import copy


class PresudoTransformer(DataFrameColumnsTransformer):
    def __init__(self, tp, columns):
        self.tp = tp
        self.columns = columns

    def fit(self, X, y=None):
        pass


class TransformerFactoryTestCase(TestCase):
    def test_factory(self):
        df = pd.DataFrame(dict(a=list(range(1000))))
        df['a'] = df.a + df.a / 10
        df['b'] = (df.a / 100).astype(int)
        df['c'] = (df.a / 10).astype(int)
        df['x'] = df.a
        tfac = (DataFrameTransformerFactory()
                .with_filter(lambda x: x != 'x')
                .on_continuous(lambda cols: PresudoTransformer('cont', cols))
                .on_categorical(lambda cols: PresudoTransformer('cat', cols))
                .on_rich_category(12, lambda cols: PresudoTransformer('rcat', cols))
                )
        tr = tfac.fit(df).transformer_
        self.assertEqual('cont', tr.transformers[0].tp)
        self.assertListEqual(['a'], tr.transformers[0].columns)

        self.assertEqual('rcat', tr.transformers[1].tp)
        self.assertListEqual(['c'], tr.transformers[1].columns)

        self.assertEqual('cat', tr.transformers[2].tp)
        self.assertListEqual(['b'], tr.transformers[2].columns)

    def test_two_factories(self):
        tfac = (
            DataFrameTransformerFactory()
            .on_continuous(lambda cols: ContinousTransformer(cols))
        )
        df1 = pd.DataFrame(dict(a=np.arange(0, 1, 0.1)))
        df2 = pd.DataFrame(dict(a=np.arange(0, 1, 0.1), b=np.arange(0, 1, 0.1)))
        tr1 = copy.deepcopy(tfac).fit(df1).transformer_
        tr2 = copy.deepcopy(tfac).fit(df2).transformer_

        self.assertEqual(1, len(tr1.transformers[0].scaler.scale_))
        self.assertEqual(2, len(tr2.transformers[0].scaler.scale_))
