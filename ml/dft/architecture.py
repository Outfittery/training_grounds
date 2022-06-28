from typing import *

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameColumnsTransformer:
    """
    Tranformer for column(s) in dataframe.
    This is not sklearn transform! It can only used within sklearn pipelines from inside of :class:``DataFrameTransformer``
    """
    def transform(self, df: pd.DataFrame) -> Iterable[Union[pd.DataFrame,pd.Series]]:
        """
        Processes the dataframe. Iteratively (with yields) return dataframes or series
        """
        raise NotImplementedError()

    def fit(self, df: pd.DataFrame) -> None:
        """
        Trains on the dataframe
        """
        raise NotImplementedError()

    def get_columns(self) -> List[str]:
        """
        Returns the columns this transformer operates on
        """
        raise NotImplementedError()


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer that processes dataframes into dataframes while preserving column names
    """
    def __init__(self, transformers: List[DataFrameColumnsTransformer]):
        self.transformers = transformers


    def fit_transform(self, df: pd.DataFrame, y=None, **kwargs) -> pd.DataFrame:
        self.fit(df,y)
        return self.transform(df)


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        buffer = []
        for transformer in self.transformers:
            for res in transformer.transform(df):
                if isinstance(res,pd.Series):
                    buffer.append(res.to_frame())
                elif isinstance(res,pd.DataFrame):
                    # result = result.merge(res, left_index=True, right_index=True, how='left') # This does not work! Sometimes in batched jobs ids are duplicated
                    buffer.append(res)
                else:
                    raise ValueError(f'DataFrameColumnsTransformer of type {type(transformer)} produced output of the unexpected type {type(res)}')
        for d in buffer:
            if not d.index.identical(df.index):
                raise ValueError(f'Something strange has happened with dataframe with columns `{",".join(c for c in d.columns[:3])}...`: the index does not match the original frame')
        if len(buffer) == 0:
            result = df[[]]
        else:
            result = pd.concat(buffer, axis=1)
        return result


    def fit(self, df: pd.DataFrame, y=None):
        for transformer in self.transformers:
            transformer.fit(df)
        return self

    def get_columns(self):
        return [c for tr in self.transformers for c in tr.get_columns()]
