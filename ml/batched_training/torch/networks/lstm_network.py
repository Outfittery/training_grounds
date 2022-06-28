from typing import *
import pandas as pd
import torch
from ....batched_training.context.architecture import AggregationFinalizer
from .network_commons import AnnotatedTensor
from .extracting_network import UniversalFactory




def lstm_data_transformation(index, contexts, df):
    names = df.index.names
    samples = list(index)
    contexts = list(contexts)
    features = list(df.columns)
    cnames = [names[1], names[0] ]
    full = pd.DataFrame([(c, s) for c in contexts for s in samples ], columns=cnames).set_index(cnames)
    full = full.merge(df.reset_index().set_index(cnames), left_index=True, right_index=True, how='left').fillna(0)
    t = torch.tensor(full.values).reshape(len(contexts), len(samples),  len(features))
    t = t.float()
    return AnnotatedTensor(t, [cnames[0], cnames[1], 'features'], [contexts, samples, features])


class LSTMFinalizer(AggregationFinalizer):
    def __init__(self, reverse_context_order = False):
        self.contexts_ = None #type: Optional[List]
        self.features_ = None #type: Optional[Tuple]
        self.reverse_context_order = reverse_context_order

    def check(self, features, aggregations):
        if len(features)!=1:
            raise ValueError('LSTMFinalizer requires exactly one featurizer')
        if len(aggregations)!=0:
            raise ValueError('LSTMFinalizer requires that no aggregators are supported. LSTMFinalizer does aggregation itself')
        return features[list(features)[0]]

    def fit(self, index: pd.DataFrame, features: Dict[str, pd.DataFrame],  aggregations: Dict[str,pd.DataFrame]):
        df = self.check(features, aggregations)
        names = df.index.names
        contexts = df.reset_index()[names[1]].unique()
        self.contexts_ = list(sorted(contexts))
        if self.reverse_context_order:
            self.contexts_ = list(reversed(self.contexts_))
        self.features_ = tuple(df.columns)

    def finalize(self, index: pd.DataFrame,  features: Dict[str, pd.DataFrame],  aggregations: Dict[str, pd.DataFrame]):
        df = self.check(features, aggregations)
        features = tuple(df.columns)
        if features!=self.features_:
            raise ValueError(f'Columns of the dataframe has changed order. Original order:\n{self.features_}\nReceived order:\n{features}')
        result = lstm_data_transformation(
            index.index,
            self.contexts_,
            df
        )
        return result


class LSTMNetwork(torch.nn.Module):
    def __init__(self, input: torch.Tensor, size: int):
        super(LSTMNetwork, self).__init__()
        if isinstance(size, list):
            if len(size)==1:
                size = size[0]
            else:
                raise ValueError(f'LSTM network is requested with sizes {size}. Size must be int, or a list, containing a single int')
        self.lstm = torch.nn.LSTM(
            input.shape[2],
            size
        )

    def forward(self, input):
        lstm_output = self.lstm(input)
        output = lstm_output[1][0]
        output = output.reshape(output.shape[1], output.shape[2])
        return output

    @staticmethod
    def Factory(size: int):
        return UniversalFactory(LSTMNetwork, 'input', 'L'+str(size), size=size)
