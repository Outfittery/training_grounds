import pandas as pd

from tg.common.ml.batched_training.context import FoldingTransformer
from unittest import TestCase

class FoldingTransformerTestCase(TestCase):
    def test_folding(self):
        df = pd.DataFrame(dict(
            sample = [0,  0,  0,  0,  0,  1,  1,  1],
            offset = [-2, -1, 0,  1,  2, -2,  0,  1],
            feature = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )).set_index(['sample','offset'])
        t = FoldingTransformer()
        rdf = t(df).reset_index()
        result = {k:list(rdf[k]) for k in rdf.columns}
        self.assertDictEqual(
            {'sample': [0, 0, 0, 1, 1, 1],
             'offset': [0, 1, 2, 0, 1, 2],
             'feature_positive': [0.3, 0.4, 0.5, 0.7, 0.8, 0.0],
             'feature_negative': [0.3, 0.2, 0.1, 0.7, 0.0, 0.6]
             }, result
        )

