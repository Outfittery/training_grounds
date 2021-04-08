from typing import *

import pandas as pd

from .architecture import DataFrameTransformer
from .column_transformers import DataFrameColumnsTransformer



class DataFrameTransformerFactory:
    """
    This class generated ``DataFrameTransformer`` by a provided rules
    """
    def __init__(self):
        self.feature_filter = None # type: Optional[Callable]
        self.feature_allow_list = None #type: Optional[List]
        self.feature_block_list = None #type: Optional[List]

        self.continuous_factory = None # type: Optional[Callable]
        self.categorical_factory = None # type: Optional[Callable]

        self.categorical_rich_factory = None # type:Optional[Callable]
        self.categorical_rich_threshold = None #type: Optional[int]

    def with_filter(self, filter: Callable) -> 'DataFrameTransformerFactory':
        """
        Specifies the condition for a column to be a feature
        """
        self.feature_filter = filter
        return self

    def with_feature_allow_list(self, allow_list: List) -> 'DataFrameTransformerFactory':
        """
                Specifies the list of acceptable features
        """
        self.feature_allow_list = allow_list
        return self

    def with_feature_block_list(self, block_list: List) -> 'DataFrameTransformerFactory':
        """
                 Specifies the list of UNacceptable features
         """

        self.feature_block_list = block_list
        return self


    def on_continuous(self, factory: Callable[[List[str]],DataFrameColumnsTransformer])-> 'DataFrameTransformerFactory':
        """
        Specifies the factory that will produce ``DataFrameColumnsTransformer`` for the continuous features.
        """
        self.continuous_factory = factory
        return self


    def on_categorical(self, factory: Callable[[List[str]],DataFrameColumnsTransformer])-> 'DataFrameTransformerFactory':
        """
        Specifies the factory that will produce ``DataFrameColumnsTransformer`` for the categorical features
        """
        self.categorical_factory = factory
        return self


    def on_rich_category(self, threshold: int, factory: Callable[[List[str]],DataFrameColumnsTransformer])-> 'DataFrameTransformerFactory':
        """
        Specifies the factory that will produce ``DataFrameColumnsTransformer`` for the categorical features that have more than ``threshold`` values

        """
        self.categorical_rich_threshold = threshold
        self.categorical_rich_factory = factory
        return self

    def _create_transformer(self, df: pd.DataFrame) -> 'DataFrameTransformer':
        transformers = []

        features = [c for c in df.columns]
        if self.feature_filter is not None:
            features = [c for c in features if self.feature_filter(c)]

        if self.feature_allow_list is not None:
            features = [c for c in features if c in self.feature_allow_list]

        if self.feature_block_list is not None:
            features = [c for c in features if c not in self.feature_block_list]

        types = df[features].dtypes
        continuous = list(types.loc[(types == 'float32') | (types == 'float64')].index)
        if len(continuous)>0:
            if self.continuous_factory is None:
                raise ValueError(f"Continuous features are presenting, but the factory is not set. Features are {continuous}")
            transformers.append(self.continuous_factory(continuous))

        categorical = [c for c in types.index if c not in continuous]
        if len(categorical)>0:
            if self.categorical_rich_threshold is not None:
                cat_value_count = {c: len(df[c].unique()) for c in categorical}
                rich_categorical = [key for key, value in cat_value_count.items() if value >= self.categorical_rich_threshold]
                if self.categorical_rich_factory is None:
                    raise ValueError(f"Rich categorical features are presenting, but the factory is not set. Features are {rich_categorical}")
                transformers.append(self.categorical_rich_factory(rich_categorical))
                categorical = [c for c in categorical if c not in rich_categorical]


        if len(categorical)>0:
            if self.categorical_factory is None:
                raise ValueError(f"Categorical features are presenting, but factory is not set. Features are {categorical}")
            transformers.append(self.categorical_factory(categorical))

        return DataFrameTransformer(transformers)


    def fit(self, X, y=None):
        self.transformer_ = self._create_transformer(X)
        self.transformer_.fit(X, y)
        return self

    def transform(self, X):
        return self.transformer_.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
