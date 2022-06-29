from unittest import TestCase
from tg.common.test_common.test_ml.test_batched.pythorch_sample_creator import *


class PytorchTestCase(TestCase):
    def test_pytorch_binary_classification(self):
        bundle = create_bundle()
        task = create_task()
        env = InMemoryTrainingEnvironment()
        task.run_with_environment(bundle, env)
        result = env.result
        self.assertGreater(result['metrics']['roc_auc_score_test'], 0.7)
        self.assertEqual(10, len(task.history))

    def test_get_batch(self):
        bundle = create_bundle()
        task = create_task()
        batch = task.generate_sample_batch(bundle, 0)
        self.assertSetEqual({'index', 'features', 'targets'}, set(batch))

    def test_get_metrics(self):
        bundle = create_bundle()
        task = create_task()
        self.assertSetEqual(
            {'roc_auc_score_train', 'roc_auc_score_test', 'roc_auc_score_display', 'loss', 'iteration'},
            set(task.get_metric_names())
        )

    def test_minibatches(self):
        bundle = create_bundle()
        task = create_task()
        task.batcher.batch_size = 100
        task.settings.mini_batch_size = 20
        task.settings.mini_epoch_count = 3
        env = InMemoryTrainingEnvironment()
        task.run_with_environment(bundle, env)
        print(env.result['metrics']['roc_auc_score_test'])
        # for s in env.message_buffer:
        #    print(s)
