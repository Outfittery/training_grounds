from tg.common.ml.batched_training.plain_extractor import *
from tg.common.ml import dft
from unittest import TestCase

INDEX = pd.DataFrame(dict(
    ind=[10,20,30, 40],
    A = [100,200,300,400],
    B = [1, 1, 2, 2]
)).set_index('ind')

A = pd.DataFrame(dict(
    A = [100, 200, 300],
    X = ['a', 'b', 'a']
)).set_index('A')

X = pd.DataFrame(dict(
    X = ['a','b'],
    Y = ['A', 'B']
)).set_index('X')

B = pd.DataFrame(dict(
    A=[100,200,300, 100],
    B=[1, 1, 2, 2],
    Z=['q','w','e','r']
)).set_index(['A','B'])

ALT_INDEX = pd.DataFrame(dict(
    ind=[10, 20, 30, 40],
    V = [-1, -2, -3, -4]
)).set_index('ind')

bundle = DataBundle(index=INDEX,a=A, x=X, b=B, alt = ALT_INDEX)
ind_small = INDEX.iloc[:3]
ind_full = INDEX

class PlainExtractorTestCase(TestCase):
    def exec_extract(self, extractor, index):
        ibundle = IndexedDataBundle(index,bundle)
        extractor.fit(ibundle)
        df = extractor.extract(ibundle)
        return df

    def run_test(self, extractor, index, columns, **result_columns):
        df = self.exec_extract(extractor, index)
        self.assertListEqual(list(index.index), list(df.index))
        self.assertListEqual(columns, list(df.columns))
        for key, value in result_columns.items():
            vs = [c if str(c)!='nan' else 'NONE' for c in df[key]]
            self.assertListEqual(value, vs)

    def test_simple(self):
        self.run_test(
            PlainExtractor.build('test').index().apply(),
            ind_small,
            ['A', 'B']
        )

    def test_column_limitation(self):
        self.run_test(
            PlainExtractor.build('test').index().apply(take_columns='A'),
            ind_small,
            ['A']
        )

    def test_join(self):
        self.run_test(
            PlainExtractor.build('test').index().join('a', 'A').apply(),
            ind_small,
            ['X'],
            X=['a','b','a']
        )

    def test_2_join(self):
        self.run_test(
            PlainExtractor.build('test').index().join('a', 'A').join('x', 'X').apply(),
            ind_small,
            ['Y'],
            Y = ['A','B','A']
        )

    def test_missing_rows(self):
        self.assertRaises(
            ValueError,
            lambda: self.exec_extract(
                PlainExtractor.build('test').index().join('a', 'A').apply(),
                ind_full
        ))

    def test_missing_rows_fix(self):
        self.run_test(
            PlainExtractor.build('test').index().join('a', 'A').apply(raise_if_rows_are_missing=False, raise_if_nulls_detected=False),
            ind_full,
            ['X'],
            X=['a','b','a','NONE']
        )

    def test_join_on_two_rows(self):
        self.run_test(
            PlainExtractor.build('test').index().join('b', ['A','B']).apply(),
            ind_small,
            ['Z'],
            Z=['q','w','e']
        )

    def test_with_transformer(self):
        transformer = dft.DataFrameTransformerFactory.default_factory()
        self.run_test(
            PlainExtractor.build('test').index().join('a', 'A').apply(transformer=transformer),
            ind_small,
            ['X_a','X_b'],
            X_a=[True,False,True],
            X_b=[False,True,False]
        )

    def test_with_non_typical_indexing(self):
        frame = pd.DataFrame(dict(ind=['a', 'b', 'c'], a=[30, 10, 20], b=[100,200,300])).set_index('ind')
        extractor = PlainExtractor.build('test').apply(take_columns=['b'])
        extractor.fit(IndexedDataBundle(frame, bundle))
        df = extractor.extract(IndexedDataBundle(frame, bundle))
        self.assertListEqual(['a', 'b', 'c'], list(df.index))
        self.assertListEqual(['b'], list(df.columns))
        self.assertListEqual([100,200,300], list(df.b))

    def test_with_non_typical_indexing_and_joins(self):
        frame = pd.DataFrame(dict(ind=['a','b','c'], a=[30,10,20])).set_index('ind')
        extractor = PlainExtractor.build('test').join('index','a').join('a','A').apply()
        extractor.fit(IndexedDataBundle(frame, bundle))
        df = extractor.extract(IndexedDataBundle(frame, bundle))
        self.assertListEqual(['a','b','c'], list(df.index))
        self.assertListEqual(['X'], list(df.columns))
        self.assertListEqual(['a','a','b'], list(df.X))


    def test_with_alt_index(self):
        self.run_test(
            PlainExtractor.build('test').index('alt').apply(),
            ind_small,
            ['V'],
            V=[-1,-2,-3]
        )

    def test_with_custom_index(self):
        extractor = PlainExtractor.build('test').index().apply()
        frame = pd.DataFrame(dict(A=[1,2]))
        result = extractor.extract(IndexedDataBundle(frame, DataBundle()))
        self.assertListEqual([1,2], list(result.A))


