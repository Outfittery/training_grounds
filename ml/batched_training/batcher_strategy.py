from typing import *

import math
import numpy as np
import pandas as pd



class BatcherStrategy:
    def get_batch_count(self, batch_size: int, df: pd.DataFrame) -> int:
        raise NotImplementedError()

    def get_batch(self, batch_size: int, df: pd.DataFrame, batch_index: int) -> pd.Index:
        raise NotImplementedError()


class SimpleBatcherStrategy(BatcherStrategy):
    def get_batch_count(self, batch_size: int, df: pd.DataFrame) -> int:
        return int(math.ceil(df.shape[0] / batch_size))

    def get_batch(self, batch_size: int, df: pd.DataFrame, batch_index: int) -> pd.Index:
        index = df.index[batch_size * batch_index:batch_size * (batch_index + 1)]
        return index


class PriorityRandomBatcherStrategy(BatcherStrategy):
    def __init__(self, priority_column: Optional[str] = None, dataset_size_factor: float = 1.0,
                 random_state: Optional[int] = None):
        self.priority_column = priority_column
        self.dataset_size_factor = dataset_size_factor
        self.random_state = random_state

    def get_batch_count(self, batch_size: int, df: pd.DataFrame) -> int:
        return int(math.ceil(self.dataset_size_factor * df.shape[0] / batch_size))

    def get_batch(self, batch_size: int, df: pd.DataFrame, batch_index: int) -> pd.Index:
        if self.priority_column is None:
            probs = pd.Series([1 for _ in range(df.shape[0])])
        else:
            probs = df[self.priority_column]
        probs = probs / probs.sum()
        idx = list(np.random.choice(
            list(range(len(df.index))),
            batch_size,
            True,
            probs
        ))
        return df.index[idx]

    @staticmethod
    def make_priorities_for_even_representation(df: pd.DataFrame, column: Any, magnitude=1):
        priorities = np.power(1 / df.groupby(column).size().to_frame('priority'), magnitude)
        prios = df[[column]]
        prios = prios.merge(priorities, left_on=column, right_index=True)
        return prios.priority
