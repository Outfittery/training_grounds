from unittest import TestCase
import pandas as pd
from tg.common.ml.batched_training.context import NewEmbeddingExtractor, ExistingEmbeddingExtractor
from tg.common.ml.batched_training import PlainExtractor, IndexedDataBundle, DataBundle

index = pd.DataFrame(dict(
    nf = ['a','b','c','a','b','a','d']
))

vocab = pd.DataFrame(dict(
    nf=['<unk>','f','e','a','b'],
    v1=[-1, 1,2,3,4],
    v2=[-11, 11,12,13,14]
)).set_index('nf')

def extractor():
    return PlainExtractor.build('nf').index().apply(take_columns='nf')

idb = IndexedDataBundle(index, DataBundle(index=index))

class EmbeddingExtractorTestCase(TestCase):
    def test_extractor_big_vocab(self):
        emb = NewEmbeddingExtractor(
            extractor(),
            6
        )
        emb.fit(idb.change_index(idb.index_frame.iloc[:6]))
        result = emb.extract(idb)
        self.assertListEqual([1,2,3,1,2,1,0], list(result.nf))

    def test_extractor_small_vocab(self):
        emb = NewEmbeddingExtractor(
            extractor(),
            3
        )
        emb.fit(idb.change_index(idb.index_frame.iloc[:6]))
        result = emb.extract(idb)
        self.assertListEqual([1,2,0,1,2,1,0], list(result.nf))

    def test_extractor_existing_vocab(self):
        emb = ExistingEmbeddingExtractor(
            extractor(),
            vocab,
            '<unk>'
        )
        emb.fit(idb.change_index(idb.index_frame.iloc[:6]))
        result = emb.extract(idb)

        self.assertListEqual([3,4,0,3,4,3,0], list(result.nf))


