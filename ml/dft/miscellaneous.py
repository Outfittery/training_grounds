import pandas as pd
import numpy as np

from sklearn.impute import MissingIndicator
from sklearn.preprocessing import OneHotEncoder

from ..._common import TGWarningStorage



class MissingIndicatorWithReporting(MissingIndicator):
    """
    An upgrade for the sklearn MissingIndicator, that generates warning instead of throwing when sees the None in
    the column that didn't have nones before
    """
    def __init__(self, missing_values=np.nan, sparse="auto"):
        super(MissingIndicatorWithReporting, self).__init__(missing_values,"missing-only",sparse,False)
        self._warnings = []

    def fit_transform(self, X, y=None):
        return super(MissingIndicatorWithReporting, self).fit_transform(X,y)

    def transform(self, X):
        if isinstance(X,pd.DataFrame):
            has_nones = X.isnull().any(axis=0)
            unexpected_nones = []
            for column_index, has_none in enumerate(has_nones):
                if has_none and column_index not in self.features_:
                    unexpected_nones.append(X.columns[column_index])
            if len(unexpected_nones)>0:
                for column in unexpected_nones:
                    TGWarningStorage.add_warning('Unexpected None', dict(reporter='MissingIndicatorWithReporting'), dict(column=column))
        return super(MissingIndicatorWithReporting, self).transform(X)



class OneHotEncoderForDataframe(OneHotEncoder):
    def __init__(self):
        super(OneHotEncoderForDataframe, self).__init__(sparse=False)

    def fit(self, X, y=None):
        X = X.values.reshape(-1,1)
        return super(OneHotEncoderForDataframe, self).fit(X,y)

    def transform(self, X):
        column_name = X.name
        values = X.values
        values = values.reshape(-1,1)
        columns = [column_name+'_'+str(cat) for cat in self.categories_[0]]
        features = super(OneHotEncoderForDataframe, self).transform(values)
        df = pd.DataFrame(features,columns=columns,index=X.index)
        return df
