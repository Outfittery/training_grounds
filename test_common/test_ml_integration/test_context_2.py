from tg.common.ml.batched_training import context as btc
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import sandbox as bts
from unittest import TestCase
from pathlib import Path


class AlternativeTestCase(TestCase):
    def run_test(self,  network_type, enable_units = None):
        task = bts.AlternativeTrainingTask2()
        if enable_units is not None:
            task.assembly_point.enable_units(enable_units)
        task.settings.epoch_count = 1
        task.settings.batch_size = 20000
        task.settings.mini_epoch_count = 5
        task.optimizer_ctor.type='torch.optim:Adam'
        task.assembly_point.network_factory.network_type = network_type
        result = task.run(bts.TestBundles.get_test_2_bundle())
        self.assertGreater(result['metrics']['roc_auc_score_test'], 0.7)


    def test_alon_attention(self):
        self.run_test(btc.Dim3NetworkType.AlonAttention, ['features'])

    def test_alon_attention_encoding_only(self):
        self.run_test(btc.Dim3NetworkType.AlonAttention, ['stem_embedding', 'ending_embedding'])

    def test_alon_attention_everything(self):
        self.run_test(btc.Dim3NetworkType.AlonAttention)

    def test_lstm(self):
        self.run_test(btc.Dim3NetworkType.LSTM)

    def test_lstm_with_attention(self):
        self.run_test(btc.Dim3NetworkType.SelfAttentionAndLSTM)

    def test_alon_attention_sigmoid(self):
        self.run_test(btc.Dim3NetworkType.AlonAttentionSigmoid)




