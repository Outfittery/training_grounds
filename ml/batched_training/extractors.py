from typing import *

import copy
import pandas as pd

from yo_fluq_ds import KeyValuePair

from .data_bundle import DataBundle, IndexedDataBundle
from ..._common import Logger


class _ExtractorWithDisabledFit:
    def __init__(self, extractor: 'Extractor'):
        self.extractor = extractor

    def fit(self, ibundle: IndexedDataBundle):
        Logger.info(f'Fitting of extractor {self.get_name()} is disabled')

    def extract(self, ibundle: IndexedDataBundle) -> pd.DataFrame:
        return self.extractor.extract(ibundle)

    def preprocess_bundle(self, bundle: IndexedDataBundle):
        return self.extractor.preprocess_bundle(bundle)

    def get_name(self):
        return self.extractor.get_name()

    def with_disabled_fit(self):
        return self


class Extractor:
    def fit(self, ibundle: IndexedDataBundle):
        raise NotImplementedError()

    def extract(self, ibundle: IndexedDataBundle) -> pd.DataFrame:
        raise NotImplementedError()

    def preprocess_bundle(self, bundle: IndexedDataBundle):
        pass

    def with_disabled_fit(self):
        return _ExtractorWithDisabledFit(self)

    def get_name(self):
        raise NotImplementedError()

    def fit_extract(self, ibundle: IndexedDataBundle):
        self.fit(ibundle)
        return self.extract(ibundle)

    @staticmethod
    def make_extraction(ibundle: IndexedDataBundle, extractors: List['Extractor']) -> IndexedDataBundle:
        result = DataBundle()
        for extractor in extractors:
            try:
                rs = extractor.extract(ibundle)
            except Exception as e:
                raise ValueError(f'Error when extracting from extractor `{extractor.get_name()}`') from e
            if isinstance(rs,dict):
                for key, value in rs.items():
                    result[key] = value
            else:
                result[extractor.get_name()] = rs
        return IndexedDataBundle(ibundle.index_frame, result)


class UnionExtractor(Extractor):
    def __init__(self, extractors: List[Extractor]):
        self.extractors = extractors
        self.name = ','.join(extractor.get_name() for extractor in self.extractors)

    def fit(self, ibundle: IndexedDataBundle):
        for extractor in self.extractors:
            extractor.fit(ibundle)

    def extract(self, ibundle: IndexedDataBundle) -> pd.DataFrame:
        return {extractor.get_name():extractor.extract(ibundle) for extractor in self.extractors}

    def get_name(self):
        return self.name


class CombinedExtractor(Extractor):
    def __init__(self, name: str, extractors: List[Extractor]):
        self.name = name
        self.extractors = extractors

    def fit(self, ibundle: IndexedDataBundle):
        for extractor in self.extractors:
            extractor.fit(ibundle)

    @staticmethod
    def _run_extractors(ibundle: IndexedDataBundle, extractors):
        frames = []
        for extractor in extractors:
            frame = extractor.extract(ibundle)
            prefix = extractor.get_name()
            if prefix is not None and prefix!='':
                prefix+='_'
                frame.columns = [prefix + c for c in frame.columns]
            frames.append(frame)
        df = pd.concat(frames, axis=1)
        return df

    def extract(self, ibundle: IndexedDataBundle):
        df = CombinedExtractor._run_extractors(ibundle, self.extractors)
        return df

    def get_name(self):
        return self.name
