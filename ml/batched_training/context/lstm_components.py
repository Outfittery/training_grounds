from typing import *
import pandas as pd
from .architecture import AggregationFinalizer
from ..torch import AnnotatedTensor, DfConversion



def lstm_data_transformation(index, contexts, df, conversion):
    names = df.index.names
    samples = list(index)
    contexts = list(contexts)
    features = list(df.columns)
    cnames = [names[1], names[0]]
    full = pd.DataFrame([(c, s) for c in contexts for s in samples], columns=cnames).set_index(cnames)
    full = full.merge(df.reset_index().set_index(cnames), left_index=True, right_index=True, how='left').fillna(0)
    t = conversion(full).reshape(len(contexts), len(samples), len(features))
    return AnnotatedTensor(t, [cnames[0], cnames[1], 'features'], [contexts, samples, features])



class LSTMFinalizer(AggregationFinalizer):
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


    def check(self, features, aggregations):
        if len(features) != 1:
            raise ValueError('LSTMFinalizer requires exactly one featurizer')
        if len(aggregations) != 0:
            raise ValueError('LSTMFinalizer requires that no aggregators are supported. LSTMFinalizer does aggregation itself')
        f = features[list(features)[0]]
        if self.feature_transformer is not None:
            f = self.feature_transformer(f)
        return f


    def fit(self, index: pd.DataFrame, features: Dict[str, pd.DataFrame], aggregations: Dict[str, pd.DataFrame]):
        df = self.check(features, aggregations)
        names = df.index.names
        contexts = df.reset_index()[names[1]].unique()
        self.contexts_ = list(sorted(contexts))
        if self.reverse_context_order:
            self.contexts_ = list(reversed(self.contexts_))
        self.features_ = tuple(df.columns)

    def finalize(self, index: pd.DataFrame, features: Dict[str, pd.DataFrame], aggregations: Dict[str, pd.DataFrame]):
        df = self.check(features, aggregations)
        features = tuple(df.columns)
        if features != self.features_:
            raise ValueError(f'Columns of the dataframe has changed order. Original order:\n{self.features_}\nReceived order:\n{features}')
        result = lstm_data_transformation(
            index.index,
            self.contexts_,
            df,
            self.conversion
        )
        return result