from unittest import TestCase
from tg.common.ml.single_frame_training.model_provider import CatBoostWrap
import pandas as pd
import catboost
from sklearn.pipeline import make_pipeline

class CatBoostWrapTest(TestCase):
    def test_classifier(self):
        df = pd.DataFrame(dict(
            a = ['a','b','c'],
            b = [1.2, 2.0, 3.1],
            c = [1,1,0]
        ))
        src_alg = catboost.CatBoostClassifier(silent=True,iterations=5)
        alg = make_pipeline(CatBoostWrap(src_alg),src_alg)
        alg.fit(df[['a','b']],df.c)
        self.assertListEqual(['a'],list(src_alg.get_params()['cat_features']))

