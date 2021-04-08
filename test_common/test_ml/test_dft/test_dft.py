from unittest import TestCase
from tg.common.ml.dft import *
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from yo_fluq_ds import Query
from sklearn.preprocessing import MinMaxScaler


class EasyDftTest(TestCase):
    def test_general(self):
        df_src = pd.DataFrame(dict(
            a=[1,1,1,1],
            b=[-1,0,1, None],
            c=[-1, 0,  1, None],
            d=[9,10,11, 10],
            e=[None,None,None,None],
            f=['a','b','c',None]
        ), index=['a1','a2','a3','a4'])
        tr = DataFrameTransformer([
            ContinousTransformer(['a','b','c','d'], scaler=MinMaxScaler()),
            CategoricalTransformer(['f'])
            ])
        df = tr.fit_transform(df_src)
        self.assertListEqual(['a', 'b', 'c', 'd', 'b_missing', 'c_missing','f'],list(df.columns))
        self.assertListEqual([0, 0, 0, 0], list(df.a))
        self.assertListEqual([0, 0.5, 1, 0.5], list(df.b))
        self.assertListEqual([0, 0.5, 1, 0.5], list(df.c))
        self.assertListEqual([0, 0.5, 1, 0.5], list(df.d))
        self.assertListEqual([False,False,False,True], list(df.b_missing))
        self.assertListEqual([False, False, False, True], list(df.c_missing))
        self.assertListEqual(['a','b','c','NONE'],list(df.f))
        self.assertListEqual(list(df_src.index),list(df.index))
        self.assertEqual(0, len(TGWarningStorage.get_report()))



    def make_test(self, train, test, transformer: DataFrameColumnsTransformer,expected, warnings:Optional[List] = None):
        df_train = pd.DataFrame(dict(a=train))
        transformer.columns = ['a']
        tr = DataFrameTransformer([transformer])
        tr.fit(df_train)
        df_test = pd.DataFrame(dict(a=test))
        if expected is None:
            print(tr.transform(df_test))
            print(pd.DataFrame(TGWarningStorage.get_report()))
            return
        elif isinstance(expected,type):
            self.assertRaises(expected,lambda:tr.transform(df_test))
        else:
            result = tr.transform(df_test)
            if callable(expected):
                self.assertTrue(expected(result))
            else:
                self.assertListEqual(expected,list(result['a']))
            if warnings is None:
                report = TGWarningStorage.get_report()
                if len(report)>0:
                    self.fail(f'Tranform had warnings, but those were not mentioned by test:\n{pd.DataFrame(report)}')
            else:
                warns = [w['_message'] for w in TGWarningStorage.get_report()]
                self.assertListEqual(warnings,warns)
            TGWarningStorage.clear()



    def test_none_column(self):
        self.make_test(
            [None,None,None],
            [1,2],
            ContinousTransformer([]),
            lambda df: len(df.columns)==0
        )

    def test_unexpected_none(self):
        self.make_test(
            [1,2,3],
            [1,None],
            ContinousTransformer([],scaler=MinMaxScaler()),
            [0.0,0.5],
            [
                'Unexpected None'
            ]
        )

    def test_categorical(self):
        self.make_test(
            [1,None,2,1],
            [1,2,None,3],
            CategoricalTransformer([]),
            ['1.0','2.0','NONE','1.0'],
            [
                'Unexpected value'
            ]
        )

    def test_extracting_warnings(self):
        df = Query.en(range(10)).select(lambda z: (z,z)).to_dataframe(columns=['x','y'])
        pipe = make_pipeline(
            DataFrameTransformer([ContinousTransformer(['x'])]),
            LinearRegression()
        )
        pipe.fit(df[['x']],df.y)
        pipe.predict(pd.DataFrame(dict(x=[None])))
        warnings = TGWarningStorage.get_report()
        self.assertEqual(1,len(warnings))
        TGWarningStorage.clear()

    def test_categorical_with_one_hot(self):
        df = pd.DataFrame(dict(x=['a','b','c'], y=['x','x','y']))
        tr = DataFrameTransformer([
            CategoricalTransformer(['x','y'],postprocessor=OneHotEncoderForDataframe())
        ])
        r = tr.fit_transform(df)
        self.assertListEqual(
            ['x_a','x_b','x_c','y_x','y_y'],
            list(r.columns)
        )
        self.assertListEqual([1,0,0],list(r.x_a))
        self.assertListEqual([0, 1, 0], list(r.x_b))
        self.assertListEqual([0, 0, 1], list(r.x_c))
        self.assertListEqual([1,1,0],list(r.y_x))
        self.assertListEqual([0, 0, 1], list(r.y_y))
        self.assertEqual(0, len(TGWarningStorage.get_report()))

    def test_top_k_strategy(self):
        df = pd.DataFrame(dict(x=['a', 'b', 'c', 'a', 'b']))
        tr = DataFrameTransformer([
            CategoricalTransformer(['x'], replacement_strategy=TopKPopularStrategy(2, 'OTHER'),
                                   postprocessor=OneHotEncoderForDataframe())
        ])
        result = tr.fit_transform(df)
        self.assertEqual(['x_OTHER','x_a','x_b'], list(result.columns))

    def test_top_k_strategy_throws_on_too_few_categories(self):
        df = pd.DataFrame(dict(x=['a','b','c']))
        tr = DataFrameTransformer([
            CategoricalTransformer(['x'],replacement_strategy=TopKPopularStrategy(3,'OTHER'),postprocessor=OneHotEncoderForDataframe())
        ])
        self.assertRaises(ValueError,lambda: tr.fit_transform(df))



