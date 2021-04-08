

from unittest import TestCase
from tg.common.test_common.test_ml.test_batched.pythorch_sample_creator import *




class PytorchTestCase(TestCase):
    def test_pytorch_binary_classification(self):
        bundle = create_bundle()
        task = create_task()
        result = task.run(bundle)
        self.assertGreater(result['metrics']['roc_auc_score_test'], 0.7)
        self.assertEqual(10, len(task.history))



