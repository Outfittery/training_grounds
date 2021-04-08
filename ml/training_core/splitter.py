from typing import *

import copy
import pandas as pd

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split



class DataFrameSplit:
    TRAIN_NAME = 'train'

    def __init__(self,
                 df: pd.DataFrame,
                 features: List,
                 labels: Any):
        self.df = df
        self.features = features
        self.labels = labels
        self.train = df.index
        self.tests = {}
        self.info = {}

    def get_xy(self, index) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (
            self.df.loc[index][self.features],
            self.df.loc[index][self.labels]
        )

    def clone(self) -> 'DataFrameSplit':
        c = DataFrameSplit(self.df, self.features, self.labels)
        c.train = copy.deepcopy(self.train)
        c.tests = copy.deepcopy(self.tests)
        c.info = copy.deepcopy(self.info)
        return c


class Splitter:
    """
    Abstract class for splitter, the entity that separates one DataFrameSplit into several
    """

    def __call__(self, dfs: DataFrameSplit) -> List[DataFrameSplit]:
        raise NotImplementedError()

    def _get_subset_names_internal(self) -> List[str]:
        raise NotImplementedError()

    def get_subset_names(self) -> List[str]:
        return list(set(self._get_subset_names_internal() + [DataFrameSplit.TRAIN_NAME]))


class FoldSplitter(Splitter):
    """
    This splitter implements basic K-fold validation splits
    """

    def __init__(self,
                 fold_count=1,
                 test_size: float = 0.3,
                 test_name: str = 'test',
                 custom_split_column: Optional[str] = None,
                 decorate=False
                 ):
        """
        Args:
            fold_count: amount of folds
            test_size: relative test_size from 0 to 1
            test_name: the name that will be assigned to test set (``test`` by default)
        """
        self.test_size = test_size
        self.fold_count = fold_count
        self.test_name = test_name
        self.custom_split_column = custom_split_column
        self.decorate = decorate

    def __call__(self, dfs: DataFrameSplit):
        result = []
        for i in range(self.fold_count):
            child_dfs = dfs.clone()

            split_attr = self.custom_split_column if self.custom_split_column is not None else 'index'
            train_df = dfs.df.loc[dfs.train]

            train_values = getattr(train_df, split_attr)

            train, test = train_test_split(train_values.unique(), test_size=self.test_size, random_state=i)

            if not self.decorate:
                child_dfs.train = train_df.loc[train_values.isin(train)].index
            child_dfs.tests[self.test_name] = train_df.loc[train_values.isin(test)].index

            child_dfs.info[self.test_name] = dict(fold=i, index=i, split_column=split_attr)
            result.append(child_dfs)
        return result

    def _get_subset_names_internal(self):
        return [self.test_name]


class TimeSplit(Splitter):
    """
    This strategy implements the splitting strategy that prevents time leak

    It separates data into following splits:
    0: train: [begin;first_training_end]                      test: [first_training_end+interlag                  ; first_training_end+interlag+prediction_span]
    1: train: [begin;first_training_end+prediction_span   ]   test: [first_training_end+interlag+prediction_span  ; first_training_end+interlag+prediction_span+prediction_span]
    K: train: [begin;first_training_end+prediction_span*K]    test: [first_training_end+interlag+prediction_span*K; first_training_end+interlag+prediction_span*K+prediction_span]
    """

    def __init__(self,
                 date_column: Any,
                 first_training_end: datetime,
                 prediction_span: timedelta,
                 interlag: timedelta,
                 training_span: Optional[timedelta] = None,
                 test_name='test'
                 ):
        self.date_column = date_column
        self.first_training_end = first_training_end
        self.prediction_span = prediction_span
        self.interlag = interlag
        self.training_span = training_span
        self.test_name = test_name

    def get_test_name(self):
        return self.test_name

    class Borders:
        def __init__(self,
                     train_begin: datetime,
                     train_end: datetime,
                     test_begin: datetime,
                     test_end: datetime):
            self.train_begin = train_begin
            self.train_end = train_end
            self.test_begin = test_begin
            self.test_end = test_end

    def _generate(self, min: datetime, max: datetime):
        if self.first_training_end < min:
            raise ValueError(
                f'First_training_end {self.first_training_end} should be greater than minimal date in the set {min}')
        split = TimeSplit.Borders(
            min,
            self.first_training_end,
            self.first_training_end + self.interlag,
            self.first_training_end + self.interlag + self.prediction_span
        )
        while True:
            yield split
            split = TimeSplit.Borders(
                split.train_begin,
                split.train_end + self.prediction_span,
                split.test_begin + self.prediction_span,
                split.test_end + self.prediction_span
            )
            if split.test_begin > max:
                break

    def __call__(self, dfs: DataFrameSplit):
        dates = dfs.df.loc[dfs.train][self.date_column]
        min = dates.min()
        max = dates.max()
        result = []
        for index, border in enumerate(self._generate(min, max)):
            child_dfs = dfs.clone()

            actual_train_begin = border.train_begin
            if self.training_span is not None:
                actual_train_begin = border.train_end - self.training_span
            child_dfs.train = dates.loc[(actual_train_begin <= dates) & (dates < border.train_end)].index

            child_dfs.tests[self.test_name] = dates.loc[(border.test_begin <= dates) & (dates < border.test_end)].index
            child_dfs.info[self.test_name] = border.__dict__
            child_dfs.info[self.test_name]['index'] = index
            result.append(child_dfs)
        return result

    def _get_subset_names_internal(self):
        return [self.test_name]


class OneTimeSplit(Splitter):
    def __init__(self, date_column: str, test_fraction: float, test_name='test'):
        self.date_column = date_column
        self.test_fraction = test_fraction
        self.test_name = test_name

    def __call__(self, dfs: DataFrameSplit):
        # TODO: for compatibility only, remove when no continuation model in training
        if not hasattr(self, 'test_fraction'):
            self.test_fraction = 0.2

        child_dfs = dfs.clone()
        min_date = dfs.df[self.date_column].min()
        max_date = dfs.df[self.date_column].max()
        test_duration = (max_date - min_date) * self.test_fraction
        sep_date = max_date - test_duration

        train = dfs.df.loc[dfs.df[self.date_column] < sep_date].index
        test = dfs.df.loc[dfs.df[self.date_column] >= sep_date].index
        child_dfs.train = train
        child_dfs.tests[self.test_name] = test
        return [child_dfs]

    def get_subset_names(self):
        return [self.test_name]


class UnionSplit(Splitter):
    def __init__(self, *splitters: Splitter):
        self.splitters = splitters

    def __call__(self, dfs: DataFrameSplit):
        result = []
        for splitter in self.splitters:
            for d in splitter(dfs):
                result.append(d)
        return result

    def _get_subset_names_internal(self):
        return [c for splitter in self.splitters for c in splitter.get_subset_names()]


class IdentitySplit(Splitter):
    def __call__(self, dfs: DataFrameSplit):
        return [dfs]

    def _get_subset_names_internal(self):
        return []


class CompositionSplit(Splitter):
    def __init__(self, *splitters: Splitter):
        self.splitters = splitters

    def __call__(self, dfs):
        result = [dfs]
        for split in self.splitters:
            result_tmp = []
            for ds in result:
                for child_ds in split(ds):
                    result_tmp.append(child_ds)
            result = result_tmp
        return result

    def _get_subset_names_internal(self):
        return [c for splitter in self.splitters for c in splitter.get_subset_names()]
