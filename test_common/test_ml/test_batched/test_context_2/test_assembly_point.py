import pandas as pd
from unittest import TestCase
from tg.common.ml.batched_training.context_2 import ContextAssemblyPoint2, FeaturesAssemblyUnit, ExistingEmbeddingAssemblyUnit
from tg.common.ml.batched_training import PlainExtractor, IndexedDataBundle, Batcher
from tg.common.ml.batched_training import factories as btf
from tg.common.ml.batched_training import context as btc
import time

from tg.common.ml.dft import DataFrameTransformerFactory
from tg.common.ml.batched_training.sandbox import PlainContextBuilder
from tg.common import DataBundle

src = pd.DataFrame(dict(
    sentence_id = [1,  1,  1,  1,  1,  1, 2,  2,  2,  2],
    word_id =     [10, 11, 12, 13, 14, 15, 20, 21, 22, 23],
    word = ['it', 'was', 'a', 'good', 'day', '.', 'yes', 'it', 'was', '!']
))
src.index = list(src.word_id)
src.index.name = 'sample_id'

features = pd.DataFrame(dict(
    word_id = [10, 11, 12, 13, 14, 15, 20, 21, 22, 23],
    pos = ['PRON',	'VERB',	'DET',	'ADJ',	'NOUN',	'PUNCT', 'X', 'PRON', 'VERB', 'PUNCT'],
)).set_index('word_id')

db = DataBundle(src=src, features=features)
idb = IndexedDataBundle(src.loc[[15,23]], db)

embedding = pd.DataFrame(dict(
    word = ['unk', 'it', 'was', 'a'],
    c1 = [0, 1, 0, 0],
    c2 = [0, 0, 1, 0],
    c3 = [0, 0, 0, 1]
)).set_index('word')

def create_features_apu():
    return FeaturesAssemblyUnit(
            PlainExtractor.build('features').index().join('features', 'another_word_id').apply(
                DataFrameTransformerFactory.default_factory(),
                ['pos']
        ))

def create_embedding_apu():
    return  ExistingEmbeddingAssemblyUnit(
        PlainExtractor.build('embedding').index().join('src', 'another_word_id').apply(
            None,
            ['word']
        ),
        embedding,
        'unk'
    )

class AssemblyPointTestCase(TestCase):
    def test_features(self):
        ap = ContextAssemblyPoint2(
            [create_features_apu()],
            PlainContextBuilder(),
            4,
        )
        extractor = ap.create_extractor()
        extractor.debug = True
        batch = Batcher([extractor]).fit_extract(2, idb)

        self.assertListEqual(['features'], list(batch.bundle.data_frames))
        self.assertListEqual([4,2,6], list(batch.bundle.features.tensor.shape))
        self.assertEqual('torch.float32', str(batch.bundle.features.tensor.dtype))
        self.assertEqual((7,6), extractor.data_.feature_dfs['features'].shape)

        network_factory = ap.create_network_factory()
        network = network_factory(batch)
        self.assertIsInstance(network.networks[0], btf.InputConversionNetwork)
        self.assertIsInstance(network.networks[1], btc.LSTMNetwork)

        result = network(batch)
        self.assertListEqual([2, 20], list(result.shape))



    def test_embedding(self):
        ap = ContextAssemblyPoint2(
            [create_embedding_apu()],
            PlainContextBuilder(),
            4,
        )
        extractor = ap.create_extractor()
        extractor.debug = True
        batch = Batcher([extractor]).fit_extract(2, idb)

        self.assertListEqual(['embedding'], list(batch.bundle.data_frames))
        self.assertListEqual([4, 2, 1], list(batch.bundle.embedding.tensor.shape))
        self.assertEqual('torch.int32', str(batch.bundle.embedding.tensor.dtype))

        network_factory = ap.create_network_factory()
        network = network_factory(batch)
        result = network(batch)

        self.assertListEqual([2, 20], list(result.shape))


    def test_two_units(self):
        ap = ContextAssemblyPoint2(
            [create_features_apu(), create_embedding_apu()],
            PlainContextBuilder(),
            4,
        )
        extractor = ap.create_extractor()
        extractor.debug = True
        batch = Batcher([extractor]).fit_extract(2, idb)

        self.assertListEqual(['features', 'embedding'], list(batch.bundle.data_frames))

        self.assertListEqual([4,2,6], list(batch.bundle.features.tensor.shape))
        self.assertEqual('torch.float32', str(batch.bundle.features.tensor.dtype))
        self.assertEqual((7,6), extractor.data_.feature_dfs['features'].shape)

        self.assertListEqual([4, 2, 1], list(batch.bundle.embedding.tensor.shape))
        self.assertEqual('torch.int32', str(batch.bundle.embedding.tensor.dtype))

        network_factory = ap.create_network_factory()
        network = network_factory(batch)
        result = network(batch)

        self.assertListEqual([2, 20], list(result.shape))





