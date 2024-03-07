import pandas as pd

from tg.common.ml.dft import DatetimeTransformer, DataFrameTransformerFactory
from unittest import TestCase
from datetime import datetime, timedelta

class DatetimeTransformerTestCase(TestCase):
    def test_transformation(self):
        df = pd.DataFrame(dict(
            a = [
                datetime(2022, 3, 1),
                datetime(2022, 3, 10),
            ],
            b= [
                datetime(2022, 3, 20),
                None
            ]
        ))
        tr = DatetimeTransformer(['a','b'], False)
        tr.fit(df)
        self.assertEqual('a', tr.reference_column)

        df = tr.transform(df)[0]
        self.assertListEqual([1, 10], list(df.a_day_in_month * 31))
        self.assertListEqual([60, 69], list(df.a_day_in_year*365))
        self.assertListEqual([1, 3], list(df.a_day_in_week*7))
        self.assertListEqual([False, False], list(df.a_is_null))

        self.assertListEqual([20, 0], list(df.b_day_in_month * 31))
        self.assertListEqual([79, 0], list(df.b_day_in_year*365))
        self.assertListEqual([6, 0], list(df.b_day_in_week*7))
        self.assertListEqual([False, True], list(df.b_is_null))

        pd.options.display.width = None


    def test_transformation_with_scaling(self):
        df_train = pd.DataFrame(dict(
            a = [datetime(2022, 1, 1) + timedelta(days=2*i) for i in range(20)],
            b = [datetime(2023, 1, 1) + timedelta(days=2 * i) for i in range(20)]
        ))

        df_test = pd.DataFrame(dict(
            a=[datetime(2022, 1, 1) + timedelta(days=2 * i + 1) for i in range(20)],
            b=[datetime(2023, 1, 1) + timedelta(days=2 * i + 1) for i in range(20)]
        ))

        tr = DatetimeTransformer(['a', 'b'])
        tr.fit(df_train)

        pd.options.display.width = None
        df = tr.transform(df_test)[0]
        self.assertTrue((df.abs()<2).all().all())

    def test_combined(self):
        df = pd.DataFrame(dict(
            a=[1.0, 2.0, 3.0, 4.0, 5.0],
            b = ['a', 'b', 'c', 'b', 'a'],
            c = [datetime(2022,1,1)+timedelta(days=i) for i in range(5)]
        ))
        tr = DataFrameTransformerFactory.default_factory(enable_datetime=True)
        df = tr.fit_transform(df)
        self.assertListEqual(
            ['a', 'b_a', 'b_b', 'b_c', 'c_unix', 'c_day_in_month', 'c_day_in_week', 'c_day_in_year', 'c_is_null'],
            list(df.columns)
        )



