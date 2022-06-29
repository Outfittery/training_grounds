from typing import *
from unittest import TestCase
from tg.common.ml.batched_training import torch as btt
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import context as btc
from tg.common.ml.batched_training.torch import networks as btn
from tg.common.ml import dft
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path


def translate_to_sequential(df):
    words = [c for c in df.columns if c.startswith('word_')]
    context_length = len(words)

    cdf = df[words].unstack().to_frame('word').reset_index()
    cdf = cdf.rename(columns=dict(level_0='word_position'))
    cdf.word_position = cdf.word_position.str.replace('word_', '').astype(int)
    cdf = cdf.sort_values(['sentence_id', 'word_position'])
    cdf['word_id'] = list(range(cdf.shape[0]))
    cdf = cdf[['word_id', 'sentence_id', 'word_position', 'word']]
    cdf.index = list(cdf['word_id'])

    idf = df[['label']].reset_index()
    idf['split'] = 'display'
    idf.index.name = 'sample_id'
    bundle = bt.DataBundle(index=idf, src=cdf)
    bundle.additional_information.context_length = context_length
    return bundle


class SentenceContextBuilder(btc.ContextBuilder):
    def build_context(self, ibundle, context_size):
        df = ibundle.index_frame[['sentence_id']]
        df = df.merge(ibundle.bundle.src.set_index('sentence_id'), left_on='sentence_id', right_index=True)
        df = df[['word_position', 'word_id', 'word']]
        df = df.loc[df.word_position < context_size]
        df = df.set_index('word_position', append=True)
        return df


df = pd.read_parquet(Path(__file__).parent / 'lstm_task.parquet')
db = translate_to_sequential(df)


class TestExtractorFactory(btt.TorchExtractorFactory):
    def __init__(self, plain_context: Optional[int], lstm_context: Optional[int] = None):
        self.plain_context = plain_context
        self.lstm_context = lstm_context

    def build_context_extractor(self, context_length):
        context_extractor = btc.ContextExtractor(
            name='plain_features',
            context_size=context_length,
            context_builder=SentenceContextBuilder(),
            feature_extractor_factory=btc.SimpleExtractorToAggregatorFactory(
                bt.PlainExtractor.build('word').index().apply(transformer=dft.DataFrameTransformerFactory.default_factory(), take_columns=['word']),
                btc.PivotAggregator()
            ),
            finalizer=btc.PandasAggregationFinalizer(
                add_presence_columns=False
            ),
            debug=True
        )
        return context_extractor

    def build_lstm_extractor(self, context_length):
        context_extractor = btc.ContextExtractor(
            name='lstm_features',
            context_size=context_length,
            context_builder=SentenceContextBuilder(),
            feature_extractor_factory=btc.SimpleExtractorToAggregatorFactory(
                bt.PlainExtractor.build('word').index().apply(
                    transformer=dft.DataFrameTransformerFactory.default_factory(), take_columns=['word']),
            ),
            finalizer=btt.LSTMFinalizer()
        )
        return context_extractor

    def create_extractors(self, task, bundle) -> List[bt.Extractor]:
        extractors = []
        extractors.append(bt.PlainExtractor.build('label').index().apply(take_columns=['label']))
        if self.plain_context is not None:
            extractors.append(self.build_context_extractor(self.plain_context))
        if self.lstm_context is not None:
            extractors.append(self.build_lstm_extractor(self.lstm_context))
        return extractors


class TestNetworkFactory(btt.TorchNetworkFactory):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def create_network(self, task, input):
        if 'plain_features' in input:
            return btn.FullyConnectedNetwork.Factory([self.hidden_size], 1).prepend_extraction('plain_features').create_network(task, input)

        if 'lstm_features' in input:
            return btn.FeedForwardNetwork.Factory(
                btn.ExtractingNetwork.Factory('lstm_features'),
                btn.LSTMNetwork.Factory(self.hidden_size),
                btn.FullyConnectedNetwork.Factory([self.hidden_size, 1])
            ).create_network(task, input)


def build_task(extractor):
    task = btt.TorchTrainingTask(
        bt.TrainingSettings(epoch_count=20, batch_size=50),
        btt.TorchTrainingSettings(),
        extractor,
        TestNetworkFactory(10),
        bt.MetricPool().add_sklearn(roc_auc_score),
    )
    return task


class TorchLstmTestCase(TestCase):
    def test_plain(self):
        task = build_task(TestExtractorFactory(4))
        task.run(db)
        self.assertGreaterEqual(task.history[-1]['roc_auc_score_display'], 0.55)

    def test_plain_minibatches(self):
        task = build_task(TestExtractorFactory(4))
        task.settings.mini_epoch_count = 5
        task.settings.mini_batch_size = 5
        task.run(db)
        self.assertGreaterEqual(task.history[-1]['roc_auc_score_display'], 0.55)

    def dont_test_lstm(self):
        task = build_task(TestExtractorFactory(None, 4))
        task.run(db)
        self.assertGreaterEqual(task.history[-1]['roc_auc_score_display'], 0.55)

    def test_lstm_minibatches(self):
        task = build_task(TestExtractorFactory(None, 4))
        task.settings.mini_epoch_count = 5
        task.settings.mini_batch_size = 5
        task.run(db)
        self.assertGreaterEqual(task.history[-1]['roc_auc_score_display'], 0.55)
