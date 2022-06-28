from tg.common.ml.batched_training.plain_extractor import *
from tg.common.ml.batched_training.extractors import CombinedExtractor
from tg.common.ml import dft
from unittest import TestCase

INDEX = pd.DataFrame(dict(
    ind=[10,20,30, 40],
    A = [100,200,300,400],
    B = [1, 1, 2, 2]
)).set_index('ind')

class CombinedExtractorTestCase(TestCase):
    def test_combined_extractors(self):
        ext_a = PlainExtractor.build('name1').index().apply(take_columns=['A'])
        ext_b = PlainExtractor.build('name2').index().apply(take_columns=['B'])
        cmb = CombinedExtractor('cmb', [ext_a, ext_b])
        bundle = DataBundle(index=INDEX)
        result = cmb.extract(IndexedDataBundle(INDEX,bundle))
        self.assertListEqual(['name1_A','name2_B'], list(result.columns))