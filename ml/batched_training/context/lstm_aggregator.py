from typing import *
from .components import ContextAggregator2
from .lstm_components import lstm_data_transformation
from ..torch import DfConversion
import pandas as pd

class LSTMAggregator(ContextAggregator2):
    def __init__(self,
                 reverse_context_order=False,
                 feature_transformer: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                 conversion: Optional[Callable] = None
                 ):
        self.contexts_ = None  # type: Optional[List]
        self.features_ = None  # type: Optional[Tuple]
        self.reverse_context_order = reverse_context_order
        self.conversion = conversion
        if self.conversion is None:
            self.conversion = DfConversion.auto
        self.feature_transformer = feature_transformer

    def fit(self, index: pd.DataFrame, features_df: pd.DataFrame):
        if self.feature_transformer is not None:
            features_df = self.feature_transformer(features_df)
        names = features_df.index.names
        contexts = features_df.reset_index()[names[1]].unique()
        self.contexts_ = list(sorted(contexts))
        if self.reverse_context_order:
            self.contexts_ = list(reversed(self.contexts_))
        self.features_ = tuple(features_df.columns)

    def aggregate_context(self, index: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_transformer is not None:
            features_df = self.feature_transformer(features_df)
        features = tuple(features_df.columns)
        if features != self.features_:
            raise ValueError(
                f'Columns of the dataframe has changed order. Original order:\n{self.features_}\nReceived order:\n{features}')
        result = lstm_data_transformation(
            index.index,
            self.contexts_,
            features_df,
            self.conversion
        )
        return result

