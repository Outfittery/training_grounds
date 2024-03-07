from tg.common.ml.batched_training import context as btc
from tg.common.ml.batched_training import gorynych as btg
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import sandbox as bts
from unittest import TestCase
from pathlib import Path


class AlternativeTestCase(TestCase):
    def run_test(self,  network_type, enable_units = None, make_assertion = True):
        task = bts.AlternativeTrainingTask()

        task.settings.epoch_count = 1
        task.settings.batch_size = 20000
        task.settings.mini_epoch_count = 5
        task.optimizer_ctor.type='torch.optim:Adam'
        task.factory.context_dimensionality_reduction_type = network_type
        task.enabled_extractors = enable_units
        result = task.run(bts.TestBundles.get_test_2_bundle())
        metric = result['metrics']['roc_auc_score_test']
        if make_assertion:
            self.assertGreater(metric, 0.7)
        return metric



    def test_alon_attention_features(self):
        self.run_test(btg.Dim3NetworkType.AlonAttention, ['features'])


    def test_alon_attention_encoding_only(self):
        self.run_test(btg.Dim3NetworkType.AlonAttention, ['stem_embedding', 'ending_embedding'])

    def test_alon_attention_everything(self):
        self.run_test(btg.Dim3NetworkType.AlonAttention)

    def test_alon_attention_stems_are_not_enough(self):
        metric = self.run_test(btg.Dim3NetworkType.AlonAttention, ['stem_embedding'], False)
        self.assertLessEqual(metric, 0.7)

    def test_lstm(self):
        self.run_test(btg.Dim3NetworkType.LSTM)

    def test_lstm_with_attention(self):
        self.skipTest('Too slow')
        self.run_test(btg.Dim3NetworkType.SelfAttentionAndLSTM)

    def test_alon_attention_sigmoid(self):
        self.skipTest('Unstable')
        self.run_test(btg.Dim3NetworkType.AlonAttentionSigmoid)




