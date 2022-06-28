from typing import *
from .architecture import DataFrameColumnsTransformer
import pandas as pd
import numpy as np

class ColumnData:
    def __init__(self,
                 input_column,
                 mapping: Dict[Any, int],
                 output_columns: List[str],
                 null_index: int,
                 missing_index: int
                 ):
        self.input_column = input_column
        self.mapping = mapping
        self.output_columns = output_columns
        self.null_index = null_index
        self.missing_index = missing_index

    def transform(self, df, ignore_missing_columns):
        if self.input_column in df.columns:
            series = df[self.input_column]
        else:
            if ignore_missing_columns:
                series = pd.Series(None, df.index, dtype='object')
            else:
                raise ValueError(f"Column {self.input_column} is not in input")

        result = pd.Series(
            np.where(
                series.isnull(),
                self.null_index,
                np.where(
                    series.isin(self.mapping),
                    series.replace(self.mapping),
                    self.missing_index)
            ),
            dtype='int',
            index=df.index
        )
        return result


class CategoricalTransformer2(DataFrameColumnsTransformer):
    def __init__(self,
                 columns: List,
                 max_values: int,
                 ignore_missing_columns_on_transform = True
                 ):
        self.columns = columns
        self.max_values = max_values
        self.column_data_ = None #type: Optional[List[ColumnData]]
        self.output_column_names_ = None #type: Optional[List[str]]
        self.ignore_missing_columns_on_transform = ignore_missing_columns_on_transform

    def _fit_column(self, df, column):
        if column not in df.columns:
            if not self.ignore_missing_columns_on_transform:
                raise ValueError(f"Column {column} not in the dataframe")
            else:
                return None

        special_values = []
        max_values = self.max_values
        has_nulls = False
        has_others = False

        if df[column].isnull().any():
            max_values-=1
            has_nulls = True

        vals = df.groupby(column).size().sort_values(ascending=False).index
        if len(vals)>max_values:
            max_values-=1
            has_others = True

        vals = vals[:max_values]
        mapping = {v:i for i,v in enumerate(vals)}
        output_columns = [str(v) for v in vals]

        if has_others:
            index_other = len(output_columns)
            output_columns.append('OTHER')
        else:
            index_other = 0

        if has_nulls:
            index_nulls = len(output_columns)
            output_columns.append('NULL')
        else:
            index_nulls = index_other

        output_columns = [f'{column}_{v}' for v in output_columns]

        return ColumnData(
            column,
            mapping,
            output_columns,
            index_nulls,
            index_other
        )

    def get_columns(self) -> List[str]:
        return self.columns

    def fit(self, df: pd.DataFrame) -> None:
        self.column_data_ = [self._fit_column(df, c) for c in self.columns]
        self.column_data_ = [c for c in self.column_data_ if c is not None]
        self.output_column_names_ = []
        for c in self.column_data_:
            self.output_column_names_.extend(c.output_columns)


    def transform(self, df: pd.DataFrame) -> Iterable[Union[pd.DataFrame,pd.Series]]:
        matrix = np.zeros((df.shape[0], len(self.output_column_names_)))
        rows = list(range(df.shape[0]))
        shift = 0
        if self.column_data_ is None:
            raise ValueError('CategoricalTransformer2 was not fitted')
        for c in self.column_data_:
            if hasattr(self,'ignore_missing_columns_on_transform'):
                columns = c.transform(df, self.ignore_missing_columns_on_transform)
            else: #TODO: legacy, remove
                columns = c.transform(df, True)
            columns += shift
            matrix[rows,columns] = 1
            shift+=len(c.output_columns)
        return [pd.DataFrame(matrix, columns=self.output_column_names_, index=df.index)]



