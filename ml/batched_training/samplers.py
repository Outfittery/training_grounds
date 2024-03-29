from typing import *

import math
import numpy as np
import pandas as pd
from .data_bundle import IndexedDataBundle


class Sampler:
    def get_batch_count(self, batch_size: int, db: IndexedDataBundle) -> int:
        raise NotImplementedError()

    def get_batch_index_frame(self, batch_size: int, db: IndexedDataBundle, batch_index: int) -> pd.DataFrame:
        raise NotImplementedError()


class SequencialSampler(Sampler):
    def get_batch_count(self, batch_size: int, db: IndexedDataBundle) -> int:
        return int(math.ceil(db.index_frame.shape[0] / batch_size))

    def get_batch_index_frame(self, batch_size: int, db: IndexedDataBundle, batch_index: int) -> pd.DataFrame:
        index = db.index_frame.iloc[batch_size * batch_index:batch_size * (batch_index + 1)]
        return index


class PriorityRandomSampler(Sampler):
    def __init__(self,
                 priority_column: Optional[str] = None,
                 dataset_size_factor: float = 1.0,
                 random_state: Optional[int] = None,
                 deduplicate=True):
        self.priority_column = priority_column
        self.dataset_size_factor = dataset_size_factor
        self.random_state = random_state
        self.deduplicate = deduplicate

    def get_batch_count(self, batch_size: int, db: IndexedDataBundle) -> int:
        return int(math.ceil(self.dataset_size_factor * db.index_frame.shape[0] / batch_size))

    def get_batch_index_frame(self, batch_size: int, db: IndexedDataBundle, batch_index: int) -> pd.DataFrame:
        if self.priority_column is None:
            probs = pd.Series([1 for _ in range(db.index_frame.shape[0])])
        else:
            probs = db.index_frame[self.priority_column]
        probs = probs / probs.sum()
        idx = list(np.random.choice(
            list(range(len(db.index_frame.index))),
            batch_size,
            True,
            probs
        ))
        result = db.index_frame.iloc[idx]
        if self.deduplicate:
            result = result[~result.duplicated()]
        return result

    @staticmethod
    def make_priorities_for_even_representation(df: pd.DataFrame, column: Any, magnitude=1):
        priorities = np.power(1 / df.groupby(column).size().to_frame('priority'), magnitude)
        prios = df[[column]]
        prios = prios.merge(priorities, left_on=column, right_index=True)
        return prios.priority
