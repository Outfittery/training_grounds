import pandas as pd
from typing import *
import copy
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .architecture import DataFrameColumnsTransformer

from .miscellaneous import MissingIndicatorWithReporting
from ..._common import TGWarningStorage



class ContinousTransformer(DataFrameColumnsTransformer):
    """
    This transformer should be applied to continuous columns.
    It applies scaler, imputer and adds missing indicator to nullable columns
    """

    def __init__(self, columns: List,
                 ignore_none_columns=True,
                 scaler: Any = StandardScaler(),
                 imputer: Any = SimpleImputer(),
                 missing_indicator: Any = MissingIndicatorWithReporting()
                 ):
        """

        Args:
            columns: List of columns to process
            ignore_none_columns: if True, ignore columns which are always None in training. Otherwise, throws.
            scaler: StandardScaler or another scaler
            imputer: SimpleImputer or another imputer
            missing_indicator: MissingIndicatorWithReporting or another missing indicator
        """
        self.columns = columns
        self.scaler = copy.deepcopy(scaler)
        self.imputer = copy.deepcopy(imputer)
        self.missing_indicator = copy.deepcopy(missing_indicator)
        self.ignore_none_columns = copy.deepcopy(ignore_none_columns)

        self.columns_ = None
        self.columns_ignored_because_of_none_ = None

    def fit(self, df):
        self.columns_ = self.columns

        if self.ignore_none_columns:
            all_none_column = df[self.columns_].isnull().all(axis=0)
            all_none_column = all_none_column.loc[all_none_column]
            self.columns_ignored_because_of_none_ = list(all_none_column.index)
            self.columns_ = [c for c in self.columns_ if c not in self.columns_ignored_because_of_none_]

        if len(self.columns_) > 0:
            self.scaler.fit(df[self.columns_])
            self.imputer.fit(df[self.columns_])
            self.missing_indicator.fit(df[self.columns_])

    def transform(self, df):
        warnings = []

        subdf = pd.DataFrame([], index=df.index)
        for column in self.columns_:
            if column in df.columns:
                subdf[column] = df[column]
            else:
                TGWarningStorage.add_warning('Missing column', dict(reporter='ContinousTransformer'),
                                             dict(column=column))
                subdf[column] = None

        if len(self.columns_) > 0:
            values = self.scaler.transform(self.imputer.transform(subdf))
            yield pd.DataFrame(values, index=df.index, columns=self.columns_, dtype='float')

            missing = self.missing_indicator.transform(subdf)
            missing_column_names = [str(self.columns_[ind]) + '_missing' for ind in self.missing_indicator.features_]
            yield pd.DataFrame(missing, index=df.index, columns=missing_column_names, dtype='object')

    def get_columns(self):
        return self.columns


class ReplacementStrategy:
    """
    Interface for inputation of previously unseen categories in categorical features processor
    """

    def fit_column(self, column: pd.Series, none_replacement) -> List:
        raise NotImplementedError()

    def transform_column(self, column: pd.Series):
        raise NotImplementedError()


class MostPopularStrategy(ReplacementStrategy):
    """Replaces the previously unseen category value with the most popular one, generating a warning on the way"""

    def fit_column(self, column: pd.Series, none_replacement):
        self.values = list(column.unique())
        groups = column.to_frame('x').groupby('x').size().sort_values(ascending=False)
        self.most_popular = groups.index[0]

    def transform_column(self, column: pd.Series):
        wrong_values = column.loc[~column.isin(self.values)].unique()
        if len(wrong_values) > 0:
            for value in wrong_values:
                TGWarningStorage.add_warning('Unexpected value', dict(reporter='MostPopularStrategy'),
                                             dict(column=column.name), dict(value=value))
        return pd.Series(
            np.where(column.isin(self.values), column, self.most_popular),
            index=column.index
        )


class TopKPopularStrategy(ReplacementStrategy):
    def __init__(self, top_popular_count, other_column):
        self.top_popular_count = top_popular_count
        self.acceptable_values = []
        self.other_column = other_column

    def fit_column(self, column: pd.Series, none_replacement):
        values = column.to_frame('x').groupby('x').size().sort_values(ascending=False)
        values_count = len(values)
        if values_count <= self.top_popular_count:
            raise ValueError(
                f'For column {column.name}, there were {values_count} different values, and {self.top_popular_count} are designed for selection. There must be at least one redundant value')
        self.acceptable_values = list(values.iloc[:self.top_popular_count].index)

    def transform_column(self, column: pd.Series):
        return pd.Series(
            np.where(column.isin(self.acceptable_values), column, self.other_column),
            index=column.index
        )


class CategoricalTransformer(DataFrameColumnsTransformer):
    """
    Transformer of categorical variable.
    Replaces previously unseen values following the given strategy.
    When needed, OneHot encoding should also be placed here
    """

    def __init__(self,
                 columns: List,
                 none_replacement='NONE',
                 replacement_strategy: ReplacementStrategy = MostPopularStrategy(),
                 postprocessor: Any = None
                 ):
        self.columns = columns
        self.strategies = {}
        self.none_replacement = none_replacement
        self.replacement_strategy = copy.deepcopy(replacement_strategy)
        self.postprocessor_prototype = copy.deepcopy(postprocessor)
        self.postprocessors = {}

    def _format(self, df, column_name):
        column = df[column_name]
        if self.none_replacement in column:
            raise ValueError(f"Column {column} contains string value `{self.none_replacement}`")
        return pd.Series(np.where(column.isnull(), self.none_replacement, column.astype(str)), index=df.index,
                         dtype='object', name=column_name)

    def fit(self, df):
        self.values = {}
        for column_name in self.columns:
            column = self._format(df, column_name)
            strategy = copy.deepcopy(self.replacement_strategy)
            strategy.fit_column(column, self.none_replacement)
            self.strategies[column_name] = strategy
            modified_column = strategy.transform_column(column)
            if self.postprocessor_prototype is not None:
                self.postprocessors[column_name] = copy.deepcopy(self.postprocessor_prototype)
                self.postprocessors[column_name].fit(modified_column)

    def transform(self, df):
        for column_name in self.columns:
            if column_name not in df.columns:
                TGWarningStorage.add_warning('Missing column', dict(reporter='CategoricalTransformer'),
                                             dict(column=column_name))
                column = pd.Series(self.none_replacement, df.index, dtype='object', name=column_name)
            else:
                column = self._format(df, column_name)
            strategy = self.strategies[column_name]
            values = strategy.transform_column(column)
            values = pd.Series(
                values,
                index=df.index,
                dtype='object',
                name=column_name
            )
            if column_name in self.postprocessors:
                values = self.postprocessors[column_name].transform(values)
            yield values

    def get_columns(self):
        return self.columns
