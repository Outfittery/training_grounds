from typing import *
import copy

from yo_fluq_ds import KeyValuePair
import pandas as pd

from .data_bundle import DataBundle



class IdentityTransform:
    """
    Transformer that does not do enything
    """
    def transform(self, df):
        return df

    def fit(self,X, y =None):
        self.columns = X.columns

    def get_columns(self):
        return self.columns


class Extractor:
    def fit(self, bundle: DataBundle):
        raise NotImplementedError()

    def extract(self, index_frame: pd.DataFrame, bundle: DataBundle) -> KeyValuePair:
        raise NotImplementedError()

    def get_columns(self):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    @staticmethod
    def make_extraction(index_df: pd.DataFrame, db:DataBundle, extractors: List['Extractor']) -> Dict[str,pd.DataFrame]:
        result = {'index': index_df}
        for extractor in extractors:
            rs = extractor.extract(index_df, db)
            result[rs.key] = rs.value
        return result




class IndexExtractor(Extractor):
    def __init__(self, name: str, transformer = None, custom_column_name = None):
        self.name = name
        self.column_name = custom_column_name or name
        if not isinstance(self.column_name, list):
            self.column_name=[self.column_name]
        self.transformer = copy.deepcopy(transformer) if transformer is not None else IdentityTransform()

    def fit(self, bundle: DataBundle):
        self.transformer.fit(bundle.index_frame[self.column_name])

    def extract(self, index_frame: pd.DataFrame, bundle: DataBundle):
        df = index_frame[self.column_name]
        return KeyValuePair(self.name, self.transformer.transform(df))


class DirectExtractor(Extractor):
    def __init__(self, name: str, transformer = None, custom_dataframe_name = None, custom_index_column = None):
        self.transformer = copy.deepcopy(transformer) if transformer is not None else IdentityTransform()
        self.name = name
        self.dataframe_name = custom_dataframe_name or self.name
        self.index_column = custom_index_column or self.name


    def fit(self, bundle: DataBundle):
        self.transformer.fit(bundle.data_frames[self.dataframe_name])

    def extract(self, index_frame: pd.DataFrame, bundle: DataBundle):
        df = bundle.data_frames[self.dataframe_name]
        index_column = index_frame[self.index_column]
        missing_values = index_frame.loc[~index_column.isin(df.index)]
        if missing_values.shape[0] > 0:
            raise ValueError(f'Missing index values at transformer {self.name}.\n' + str(missing_values))
        df = df.loc[index_column]
        df = self.transformer.transform(df)
        df.index = index_column.index
        return KeyValuePair(self.name, df)

    def get_columns(self):
        return self.transformer.get_columns()

    def get_name(self):
        return self.name


class LeftJoinExtractor(Extractor):
    def __init__(self,
                 name: str,
                 index_columns: List[str],
                 dataframe_name: str,
                 transformer = None
                 ):
        self.name = name
        self.index_columns = index_columns
        self.dataframe_name = dataframe_name
        self.transformer = copy.deepcopy(transformer) if transformer is not None else IdentityTransform()

    def fit(self, bundle: DataBundle):
        self.transformer.fit(bundle.data_frames[self.dataframe_name])

    def extract(self, index_frame: pd.DataFrame, bundle: DataBundle):
        index = index_frame[self.index_columns]
        df = bundle.data_frames[self.dataframe_name]
        df = index.merge(df,left_on=self.index_columns,right_index=True, how='left')
        if df.shape[0]!=index.shape[0]:
            raise ValueError(f'Something wrong happened with the merge: Index had {index.shape[0]} rows, the result {df.shape[0]}. Are the indices unique?')
        if (df.index!=index.index).any():
            raise ValueError(f"Something wrong happened with the merge: indices are not equal!")

        df = self.transformer.transform(df)
        return KeyValuePair(self.name, df)

    def get_columns(self):
        return self.transformer.get_columns()

    def get_name(self):
        return self.name


class CombinedExtractor(Extractor):
    def __init__(self, name: str, extractors: List[Extractor]):
        self.name = name
        self.extractors = extractors

    def fit(self, bundle: DataBundle):
        for extractor in self.extractors:
            extractor.fit(bundle)

    def extract(self, index_frame: pd.DataFrame, bundle: DataBundle):
        frames = [extractor.extract(index_frame, bundle).value for extractor in self.extractors]
        df = pd.concat(frames,axis=1)
        return KeyValuePair(self.name, df)

    def get_columns(self):
        return [c for extractor in self.extractors for c in extractor.get_columns()]

    def get_name(self):
        return self.name
