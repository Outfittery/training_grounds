from unittest import TestCase
from sklearn.linear_model import LogisticRegression
from tg.common.ml.single_frame_training import ModelConstructor
from catboost import CatBoostClassifier


class ModelCtorTestCase(TestCase):
    def test_lr(self):
        settings = ModelConstructor('sklearn.linear_model:LogisticRegression')
        obj = settings()
        self.assertIsInstance(obj, LogisticRegression)

    def test_catboost(self):
        settings = ModelConstructor('catboost:CatBoostClassifier', silent=True)
        obj = settings()
        self.assertIsInstance(obj, CatBoostClassifier)
        self.assertTrue(obj.get_param('silent'))
