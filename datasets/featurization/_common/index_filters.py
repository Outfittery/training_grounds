from typing import *
import pandas as pd

class AbstractIndexFilter:
    def filter(self, df: pd.DataFrame, buffer: Any) -> Tuple[pd.DataFrame, Any]:
        raise NotImplementedError()


class EmptyIndexFilter(AbstractIndexFilter):
    def filter(self, df: pd.DataFrame, buffer: Any) -> Tuple[pd.DataFrame, Any]:
        return df, buffer


class SimpleIndexFilter(AbstractIndexFilter):
    def filter(self, df: pd.DataFrame, buffer: Any) -> Tuple[pd.DataFrame, Any]:
        if buffer is None:
            return df, df.index
        index = buffer #type: pd.Index
        new_index = index.union(df.index)
        return df.loc[~df.index.isin(index)], new_index


class PerformativeIndexFilter(AbstractIndexFilter):
    def filter(self, df: pd.DataFrame, buffer: Any) -> Tuple[pd.DataFrame, Any]:
        idf = df[[]].reset_index()
        idx = None
        for c in idf.columns:
            s = idf[c].astype(str)
            if idx is None:
                idx = s
            else:
                idx = idx+'###'+s
        idx = pd.Series(list(idx), index=df.index)
        if buffer is None:
            return df, set(idx)
        rdf = df.loc[~idx.isin(buffer)]
        buffer = buffer.union(idx)
        return rdf, buffer



