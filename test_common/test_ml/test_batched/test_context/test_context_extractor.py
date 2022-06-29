import tg.common.ml.batched_training.context as btc
import tg.common.ml.batched_training.torch as btt
import tg.common.ml.batched_training as bt
import tg.common.ml.dft as dft
from functools import partial
from unittest import TestCase
import pandas as pd
from yo_fluq_ds import *

bundle = bt.DataBundle(src=pd.DataFrame(dict(
    sentence_id=[1, 1, 1, 1],
    word_id=[1, 2, 3, 4],
    word=['a', 'b', 'c', 'a']
)))
bundle.src.index.name = 'sample_id'




class SentenceLeftContextBuilder(btc.ContextBuilder):
    def __init__(self, include_pivot):
        self.include_pivot = include_pivot

    def build_context(self, ibundle: bt.IndexedDataBundle, context_size) -> pd.DataFrame:
        src = ibundle.bundle.src
        df = ibundle.index_frame[['word_id']].rename(columns=dict(word_id='_original_word_id'))
        df = df.merge(src.set_index('word_id')[['sentence_id']], left_on='_original_word_id', right_index=True)
        df = df.merge(src.set_index('sentence_id'), left_on='sentence_id', right_index=True)
        if self.include_pivot:
            df = df.loc[df.word_id <= df._original_word_id]
        else:
            df = df.loc[df.word_id < df._original_word_id]
        df['_sample_id'] = list(df.index)
        df = df.feed(fluq.add_ordering_column('_sample_id', ('word_id', False), 'offset'))
        df = df.loc[df.offset < context_size]
        df = df.drop(['_original_word_id', '_sample_id'], axis=1)
        if not self.include_pivot:
            df['offset'] += 1
        df = df.set_index('offset', append=True)
        return df


def build_context_extractor():
    tfac = (dft
        .DataFrameTransformerFactory()
        .on_categorical(partial(
        dft.CategoricalTransformer,
        postprocessor=dft.OneHotEncoderForDataframe()
    )))
    ctx = btc.ContextExtractor(
        'test',
        2,
        SentenceLeftContextBuilder(False),
        btc.SimpleExtractorToAggregatorFactory(
            bt.PlainExtractor.build('word').apply(tfac, take_columns=['word']),
            btc.PivotAggregator()
        ),
        btc.PandasAggregationFinalizer(),
        debug=True
    )
    ctx.fit(bt.IndexedDataBundle(bundle.src, bundle))

    return ctx


class ContextExtractorTestCase(TestCase):
    def test_context_extractor(self):
        ctx = build_context_extractor()
        rdf = ctx.extract(bt.IndexedDataBundle(bundle.src.iloc[2:3], bundle))

        self.assertListEqual(
            ['f0a0_word_a_at_1', 'f0a0_word_a_at_2', 'f0a0_word_b_at_1', 'f0a0_word_b_at_2', 'f0a0_word_c_at_1', 'f0a0_word_c_at_2', 'f0a0_present_f0a0'],
            list(rdf.columns)
        )
        self.assertEqual(1, rdf.shape[0])
        self.assertListEqual(
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            list(rdf.iloc[0])
        )

    def test_context_extractor_keeps_columns(self):
        ctx = build_context_extractor()
        dfs = []
        dfs.append(ctx.extract(bt.IndexedDataBundle(bundle.src.iloc[0:1], bundle)))
        dfs.append(ctx.extract(bt.IndexedDataBundle(bundle.src.iloc[1:2], bundle)))
        dfs.append(ctx.extract(bt.IndexedDataBundle(bundle.src.iloc[2:3], bundle)))
        for df in dfs:
            self.assertListEqual(
                ['f0a0_word_a_at_1', 'f0a0_word_a_at_2', 'f0a0_word_b_at_1', 'f0a0_word_b_at_2', 'f0a0_word_c_at_1', 'f0a0_word_c_at_2', 'f0a0_present_f0a0'],
                list(df.columns)
            )

    def test_lstm_extractor(self):
        ctx = btc.ContextExtractor(
            'test',
            2,
            SentenceLeftContextBuilder(False),
            btc.SimpleExtractorToAggregatorFactory(
                bt.PlainExtractor.build('word').index().apply(dft.DataFrameTransformerFactory.default_factory(), take_columns='word')
            ),
            btt.LSTMFinalizer()
        )
        ibundle = bt.IndexedDataBundle(bundle.src, bundle)
        ctx.fit(ibundle)
        res = ctx.extract(ibundle)  # type: btt.AnnotatedTensor
        self.assertListEqual(['offset', 'sample_id', 'features'], res.dim_names)
        self.assertListEqual([2, 4, 3], list(res.tensor.shape))
