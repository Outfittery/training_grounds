from typing import *
import numpy as np
import pandas as pd

class Aggregator:
    def __add__(self, other):
        if not isinstance(other, Aggregator):
            raise ValueError("The argument must be an aggregator")
        aggregators = ()
        for ag in [self,other]:
            if isinstance(ag, CombinedAggregator):
                aggregators+=ag.aggregators
            else:
                aggregators+=(ag,)
        return CombinedAggregator(aggregators)


class CombinedAggregator(Aggregator):
    def __init__(self, aggregators):
        self.aggregators = tuple(aggregators)

    def __call__(self, obj):
        dfs = [a(obj) for a in self.aggregators]
        return pd.concat(dfs, axis=0)


class PandasAggregator(Aggregator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, obj):
        result = obj.aggregate(self.kwargs)
        if isinstance(result,pd.Series):
            result = result.to_frame()
        result.columns = ['_'.join(c) for c in result.columns]
        return result


class CustomAggregator(Aggregator):
    def __init__(self, columns, aggregator: Callable[[pd.Series, Dict], None]):
        if columns is not None and not isinstance(columns,list) and not isinstance(columns, tuple):
            columns = [columns]
        self.columns = columns
        self.aggregator = aggregator


    def _get_groupby_columns(self, grby):
        return [g.name for g in grby.grouper._groupings]

    def _populate_dict_with_keys(self, row, grby_columns, group):
        values = group
        if not isinstance(values, tuple):
            values = (values,)
        for key, value in zip(grby_columns,values):
            row[key]=value

    def _produce_df(self, rows, index):
        df = pd.DataFrame(rows)
        if index is not None:
            df = df.set_index(index)
        if df.shape[0]==0 or df.shape[1] == 0:
            return  df
        all_tuples = True
        for c in df.columns:
            if not isinstance(c, tuple):
                all_tuples = False
        if all_tuples:
            df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df



    def _apply_on_grby_df(self, grby):
        grby_columns = self._get_groupby_columns(grby)
        rows = []
        for group, df in grby:
            row = {}
            self._populate_dict_with_keys(row, grby_columns, group)
            columns = self.columns
            if columns is None:
                if grby._selection is not None:
                    columns = grby._selection
                else:
                    columns = [c for c in df.columns if c not in grby_columns]
            for c in columns:
                self.aggregator(df[c], row)
            rows.append(row)
        return self._produce_df(rows, grby_columns)

    def _apply_on_grby_series(self, grby):
        if self.columns is not None:
            raise ValueError("Can't specify columns when applying to SeriesGroupBy")
        grby_columns = self._get_groupby_columns(grby)
        rows = []
        for group, serie in grby:
            row = {}
            self._populate_dict_with_keys(row, grby_columns, group)
            self.aggregator(serie, row)
            rows.append(row)
        return self._produce_df(rows, grby_columns)

    def _apply_on_df(self, df):
        row = {}
        columns = self.columns
        if columns is None:
            columns = list(df.columns)
        for c in columns:
            self.aggregator(df[c], row)
        return self._produce_df([row], None)

    def _apply_on_serie(self, serie):
        if self.columns is not None:
            raise ValueError("Can't specify columns when applying to Series")
        row = {}
        self.aggregator(serie, row)
        return self._produce_df([row], None)

    def __call__(self, obj: Union[pd.Series, pd.DataFrame, pd.core.groupby.generic.SeriesGroupBy, pd.core.groupby.generic.DataFrameGroupBy]):
        if isinstance(obj, pd.Series):
            return self._apply_on_serie(obj)
        elif isinstance(obj, pd.DataFrame):
            return self._apply_on_df(obj)
        elif isinstance(obj, pd.core.groupby.generic.SeriesGroupBy):
            return self._apply_on_grby_series(obj)
        elif isinstance(obj, pd.core.groupby.generic.DataFrameGroupBy):
            return self._apply_on_grby_df(obj)
        else:
            raise TypeError(f"Aggregator is called upon {type(obj)}")




