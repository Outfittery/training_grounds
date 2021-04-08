from typing import *

import copy
import inspect
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from ..training_core import DataFrameSplit



class AbstractModelProvider:
    def get_model(self, dfs: DataFrameSplit) -> Any:
        raise NotImplementedError()


class ModelConstructor:
    """
    Class that describes the constructor of the class and its parameters
    """
    def __init__(self, type_name: str, **kwargs: Any):
        """
        Args:
            type_name: has format ``a.b.c:e``, where ``a.b.c`` is a Python module and ``e`` is a class located in this module
            kwargs: arguments of class's constructor. Traditional hyperparameters goes here.
        """
        self.type_name = type_name
        self.kwargs = kwargs

    def _load_semicolor_part(self, part):
        mod = __import__(part)
        subpath = part.split('.')
        for s in subpath[1:]:
            methods = inspect.getmembers(mod)
            new_mods = [obj for name, obj in methods if name == s]
            if len(new_mods) == 0:
                raise ValueError('Path {0} is not found in module {1}'.format(s, mod))
            if len(new_mods) > 1:
                raise ValueError('More than two objects at path {0} are found in module {1}'.format(s, mod))
            mod = new_mods[0]
        return mod

    def _load_dotted_part(self, mod, path):
        result = mod
        for p in path:
            result = getattr(mod, p)
        return result

    def _load_class(self, path):
        parts = path.split(':')
        mod = self._load_semicolor_part(parts[0])
        result = self._load_dotted_part(mod, parts[1:])
        return result

    def __call__(self) -> Any:
        cls = self._load_class(self.type_name)
        instance = cls(**self.kwargs)
        return instance


class ColumnNamesKeeper(BaseEstimator,TransformerMixin):
    """
    This fake sklearn "estimator" remembers the names and types of the columns when fitting
    """
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def fit(self, df: pd.DataFrame, y=None):
        self.column_types_ = df.dtypes
        self.column_names_ = list(df.columns)
        return self


class CatBoostWrap:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def fit(self, X: pd.DataFrame, y = None):
        types = X.dtypes
        categorical = types.loc[types!='float'].index
        self.algorithm.set_params(cat_features=categorical)

    def transform(self, X: pd.DataFrame):
        return X

    def fit_transform(self,X: pd.DataFrame, y = None):
        self.fit(X,y)
        return X


class ModelProvider(AbstractModelProvider):
    def __init__(self,  constructor: Callable, transformer: Optional = None, model_fix: Optional[Callable] = None, keep_column_names = True):
        self.transformer = transformer
        self.constructor = constructor
        self.model_fix = model_fix
        self.keep_column_names = keep_column_names

    def get_model(self, dfs: DataFrameSplit) -> Any:
        df = dfs.df[dfs.features].loc[dfs.train]
        if self.transformer is not None:
            transformer = copy.deepcopy(self.transformer)
        else:
            transformer = None
        instance = self.constructor()
        if self.model_fix is not None:
            instance = self.model_fix(instance)

        steps = []
        if self.keep_column_names:
            steps.append(('ColumnNamesKeeper', ColumnNamesKeeper()))
        if transformer is not None:
            steps.append(('Transformer',transformer))
            if self.keep_column_names:
                steps.append(('ColumnNamesKeeperAfterTransformation', ColumnNamesKeeper()))
        steps.append(('Model',instance))
        if len(steps)==1:
            return steps[0][1]
        else:
            return Pipeline(steps)

    @staticmethod
    def catboost_model_fix(instance):
        return Pipeline([('CategoricalVariablesSetter',CatBoostWrap(instance)),('Model',instance)])
