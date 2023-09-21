from typing import *

import pandas as pd
import numpy as np

from .architecture import DataFrameColumnsTransformer
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from datetime import datetime

class DatetimeTransformer(DataFrameColumnsTransformer):
    def __init__(self, columns: List[str], with_scaler: bool = True):
        self.columns = columns
        self.with_scaler = with_scaler
        self.reference_column = None #type: Optional[str]
        self.scaler = None #type: Optional[StandardScaler]


    def _preprocess_df(self, df):
        s = {}
        for c in self.columns:
            if not is_datetime(df[c]):
                raise ValueError(f'Column {c} is not datetime')

        s = {}
        for c in self.columns:
            s[c + '_unix'] = ((df[c] - pd.Timestamp(1970,1,1)) / pd.Timedelta('1s'))
            s[c + '_day_in_month'] = df[c].dt.day / df[c].dt.days_in_month
            s[c + '_day_in_week'] = df[c].dt.day_of_week / 7
            s[c + "_day_in_year"] = df[c].dt.day_of_year / (365 + df[c].dt.is_leap_year)
            s[c + '_is_null'] = df[c].isnull()

        for c in self.columns:
            if c == self.reference_column:
                continue
            s[c + '_delta'] = s[c + '_unix'] - s[self.reference_column + '_unix']

        return pd.DataFrame(s)

    def _postprocess_df(self, df):
        if self.with_scaler:
            values = self.scaler.transform(df)
        else:
            values = df.to_numpy()
        rdf = pd.DataFrame(values, columns=df.columns, index=df.index)
        rdf = rdf.fillna(0)
        return rdf

    def fit(self, df: pd.DataFrame) -> None:
        self.reference_column = df[self.columns].isnull().mean().sort_values().index[0]
        pdf = self._preprocess_df(df)

        if self.with_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(pdf)

    def transform(self, df: pd.DataFrame) -> Iterable[Union[pd.DataFrame, pd.Series]]:
        pdf = self._preprocess_df(df)
        return [self._postprocess_df(pdf)]




