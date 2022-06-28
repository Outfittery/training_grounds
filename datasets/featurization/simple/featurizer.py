from typing import *

import pandas as pd

from collections import Counter, defaultdict



# helper method needed to pickel defauldict
def dc():
    return defaultdict(Counter)


class StreamFeaturizer:
    def start(self) -> None:
        raise NotImplementedError()

    def observe_data_point(self, item) -> Optional[pd.DataFrame]:
        raise NotImplementedError()

    def finish(self) -> Optional[pd.DataFrame]:
        raise NotImplementedError()


class DataframeFeaturizer(StreamFeaturizer):
    def __init__(self, buffer_size: Optional[int] = None, row_selector: Optional[Callable] = None):
        self.buffer_size = buffer_size
        self.row_selector = row_selector

    def start(self) -> None:
        self.buffer = []

    def _featurize(self, item: Any) -> List[Any]:
        if self.row_selector is not None:
            return [self.row_selector(item)]
        raise NotImplementedError()

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _validate(self):
        pass

    def observe_data_point(self, item) -> Optional[pd.DataFrame]:
        rows = self._featurize(item)
        self.buffer.extend(rows)
        if self.buffer_size is not None and len(self.buffer)>=self.buffer_size:
            return self._flush()

    def _flush(self) -> pd.DataFrame:
        df = pd.DataFrame(self.buffer)
        df = self._postprocess(df)
        self.buffer = []
        return df

    def finish(self) -> Optional[pd.DataFrame]:
        self._validate()
        if len(self.buffer)>0:
            return self._flush()


class AggegatedStatsFeaturizer(StreamFeaturizer):
    def __init__(self, aggregation_levels):
        self.aggregation_levels = aggregation_levels

    def start(self):
        self.stats = {}
        for aggregation_level in self.aggregation_levels:
            self.stats[aggregation_level] = defaultdict(dc)

    def _select(self, item):
        raise NotImplementedError()

    def _extract_counts(self, row):
        raise NotImplementedError()

    def observe_data_point(self, item):
        # select relevant values of item
        row = self._select(item)
        # # extract counts for given item
        self._extract_counts(row)

    def finish(self):
        raise NotImplementedError()
