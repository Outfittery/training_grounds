from typing import *

import os
import pandas as pd

from pathlib import Path
from yo_fluq_ds import Query

from ..training_core import DataFrameSplit



class DataFrameLoader:
    """
    Default data provider, currently solving all the use cases.
    Argument can be pd.DataFrame, or path to parquet file or folder with parquet files
    """

    def __init__(self,
                 label_column,
                 feature_groups: Optional[Dict[str, List[str]]] = None,
                 sample_size: Optional[float] = None,
                 ignore_missing_features=True
                 ):
        """
        Args:
            label_column: column for labels
            feature_groups: feature columns
            sample_size: if not-None, dataframe with be sampled with this coefficient
            ignore_missing_features: if False, will throw if some feature columns are not in dataframe
        """
        self.label_column = label_column
        self.feature_groups = feature_groups
        self.disable_feature_group = {}
        self.sample_size = sample_size
        self.ignore_missing_features = ignore_missing_features

    def _load_data(self, data: Union[str, Path, pd.DataFrame]):
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, str) or isinstance(data, Path):
            data = str(data)
            if os.path.isfile(data):
                return pd.read_parquet(data)
            elif os.path.isdir(data):
                dfs = Query.folder(data).select(pd.read_parquet).to_list()
                return pd.concat(dfs, sort=False)
            else:
                raise ValueError(f'Data was `{data}`, but there is neither file nor folder on this location')
        else:
            raise ValueError(f"Data was `{data}`, but the format is not supported")

    def get_data(self, data: Union[str, Path, pd.DataFrame]):
        df = self._load_data(data)

        if self.sample_size is not None:
            df = df.sample(frac=self.sample_size)

        if self.feature_groups is None:
            features = [c for c in df.columns if c != self.label_column]
        else:
            features = []
            for key, value in self.feature_groups.items():
                if not self.disable_feature_group.get(key, False):
                    for v in value:
                        features.append(v)

        if self.ignore_missing_features:
            features = [v for v in features if v in df.columns]

        return DataFrameSplit(
            df,
            features,
            self.label_column
        )
