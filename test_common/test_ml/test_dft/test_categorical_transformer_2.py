from tg.common.ml.dft import CategoricalTransformer2
from unittest import TestCase
import pandas as pd
from yo_fluq_ds import *

def make_list_line(collection, path):
    return f"self.assertListEqual({list(collection)}, list({path}))"

def mk(df):
    cl = []
    cl.append(make_list_line(df.columns, 'df.columns'))
    cl.append(make_list_line(df.cmp_val, 'df.cmp_val'))
    print(df)
    print('\n\n')
    print('\n'.join(cl))


A = pd.DataFrame(
    dict(a=[1,2,3,1,2,1]*2,
         b=[3, 3, 2, 2, 2, 1]*2,
         c=[1,1,1,3,4,5,2,2,6,7,8,9],
         n=['1','1','1','2','2','3','4'] + [None]*5,
         nn = [None]*12,
         s=['1','2','1','3','4','5']*2
))




class CategoricalTransformer2TestCase(TestCase):
    def fit_train(self, columns, columns_count, test_on = None):
        if test_on is None:
            test_on = A
        tr = CategoricalTransformer2(columns, columns_count)
        tr.fit(A)
        df = tr.transform(test_on)[0]
        vals = []
        for row in Query.df(df):
            vs = []
            for key, v in row.items():
                if v>0.5:
                    vs.append(key)
            vals.append('/'.join(vs))
        df['cmp_val'] = vals
        return df

    #region Columns and combinations

    def test_small_column(self):
        df = self.fit_train(['a'], 5)
        self.assertListEqual(['a_1', 'a_2', 'a_3', 'cmp_val'], list(df.columns))
        self.assertListEqual(['a_1', 'a_2', 'a_3', 'a_1', 'a_2', 'a_1', 'a_1', 'a_2', 'a_3', 'a_1', 'a_2', 'a_1'], list(df.cmp_val))

    def test_two_small_columns(self):
        df = self.fit_train(['a','b'], 5)
        self.assertListEqual(['a_1', 'a_2', 'a_3', 'b_2', 'b_3', 'b_1', 'cmp_val'], list(df.columns))
        self.assertListEqual(
            ['a_1/b_3', 'a_2/b_3', 'a_3/b_2', 'a_1/b_2', 'a_2/b_2', 'a_1/b_1', 'a_1/b_3', 'a_2/b_3', 'a_3/b_2',
             'a_1/b_2', 'a_2/b_2', 'a_1/b_1'], list(df.cmp_val))

    def test_big_column(self):
        df = self.fit_train(['c'], 4)
        self.assertListEqual(['c_1', 'c_2', 'c_3', 'c_OTHER', 'cmp_val'], list(df.columns))
        self.assertListEqual(
            ['c_1', 'c_1', 'c_1', 'c_3', 'c_OTHER', 'c_OTHER', 'c_2', 'c_2', 'c_OTHER', 'c_OTHER', 'c_OTHER',
             'c_OTHER'], list(df.cmp_val))

    def test_big_and_small_columns(self):
        df = self.fit_train(['a','c'], 4)
        self.assertListEqual(['a_1', 'a_2', 'a_3', 'c_1', 'c_2', 'c_3', 'c_OTHER', 'cmp_val'], list(df.columns))
        self.assertListEqual(
            ['a_1/c_1', 'a_2/c_1', 'a_3/c_1', 'a_1/c_3', 'a_2/c_OTHER', 'a_1/c_OTHER', 'a_1/c_2', 'a_2/c_2',
             'a_3/c_OTHER', 'a_1/c_OTHER', 'a_2/c_OTHER', 'a_1/c_OTHER'], list(df.cmp_val))
    #endregion

    #region Nones

    def test_none_small_columns(self):
        df = self.fit_train(['n'], 5)
        self.assertListEqual(['n_1', 'n_2', 'n_3', 'n_4', 'n_NULL', 'cmp_val'], list(df.columns))
        self.assertListEqual(
            ['n_1', 'n_1', 'n_1', 'n_2', 'n_2', 'n_3', 'n_4', 'n_NULL', 'n_NULL', 'n_NULL', 'n_NULL', 'n_NULL'],
            list(df.cmp_val))

    def test_none_big_columns(self):
        df = self.fit_train(['n'], 4)
        self.assertListEqual(['n_1', 'n_2', 'n_OTHER', 'n_NULL', 'cmp_val'], list(df.columns))
        self.assertListEqual(
            ['n_1', 'n_1', 'n_1', 'n_2', 'n_2', 'n_OTHER', 'n_OTHER', 'n_NULL', 'n_NULL', 'n_NULL', 'n_NULL', 'n_NULL'],
            list(df.cmp_val))

    def test_none_only_column(self):
        df = self.fit_train(['nn'], 2)
        self.assertListEqual(['nn_NULL', 'cmp_val'], list(df.columns))
        self.assertListEqual(
            ['nn_NULL', 'nn_NULL', 'nn_NULL', 'nn_NULL', 'nn_NULL', 'nn_NULL', 'nn_NULL', 'nn_NULL', 'nn_NULL',
             'nn_NULL', 'nn_NULL', 'nn_NULL'], list(df.cmp_val))

    #endregion

    def test_unexpected_other_in_small_nullable(self):
        df = self.fit_train(['n'], 5, pd.DataFrame(dict(n=['1','2','X'])))
        self.assertListEqual(['n_1', 'n_2', 'n_3', 'n_4', 'n_NULL', 'cmp_val'], list(df.columns))
        self.assertListEqual(['n_1', 'n_2', 'n_1'], list(df.cmp_val))

    def test_unexpected_other_in_small_nonnullable(self):
        df = self.fit_train(['s'], 5, pd.DataFrame(dict(s=['1','2','X'])))
        self.assertListEqual(['s_1', 's_2', 's_3', 's_4', 's_5', 'cmp_val'], list(df.columns))
        self.assertListEqual(['s_1', 's_2', 's_1'], list(df.cmp_val))


    def test_unexpected_other_in_big_nullable(self):
        df = self.fit_train(['n'], 4, pd.DataFrame(dict(n=['1','2','X'])))
        self.assertListEqual(['n_1', 'n_2', 'n_OTHER', 'n_NULL', 'cmp_val'], list(df.columns))
        self.assertListEqual(['n_1', 'n_2', 'n_OTHER'], list(df.cmp_val))

    def test_unexpected_other_in_big_nonnullable(self):
        df = self.fit_train(['n'], 4, pd.DataFrame(dict(n=['1','2','X'])))
        self.assertListEqual(['n_1', 'n_2', 'n_OTHER', 'n_NULL', 'cmp_val'], list(df.columns))
        self.assertListEqual(['n_1', 'n_2', 'n_OTHER'], list(df.cmp_val))


    def test_unexpected_none_in_small(self):
        df = self.fit_train(['s'],5, pd.DataFrame(dict(s=['1','2',None])))
        self.assertListEqual(['s_1', 's_2', 's_3', 's_4', 's_5', 'cmp_val'], list(df.columns))
        self.assertListEqual(['s_1', 's_2', 's_1'], list(df.cmp_val))

    def test_unexpected_none_in_big(self):
        df = self.fit_train(['s'], 4, pd.DataFrame(dict(s=['1', '2', None])))
        self.assertListEqual(['s_1', 's_2', 's_3', 's_OTHER', 'cmp_val'], list(df.columns))
        self.assertListEqual(['s_1', 's_2', 's_OTHER'], list(df.cmp_val))

