from unittest import TestCase
from tg.common.ml.batched_training import PrecomputingExtractor, DataBundle, PlainExtractor, IndexedDataBundle
from yo_fluq_ds import *

class PrecomputingExtractorTestCase(TestCase):
    def test_precomputing_extractor(self):
        df = Query.en(range(10)).select(lambda z: dict(a=z, b=10*z, c = 100*z)).to_dataframe()
        bundle = DataBundle(index=df)
        inner_extractor = PlainExtractor.build('t').index().apply(take_columns=['a', 'b'])
        caching_extractor = PrecomputingExtractor('test', inner_extractor)
        ibundle = IndexedDataBundle(bundle.index,bundle)
        caching_extractor.preprocess_bundle(ibundle)
        self.assertListEqual(['index'], list(bundle.data_frames))
        caching_extractor.fit(ibundle)
        self.assertListEqual(['index', 'test'], list(bundle.data_frames))
        self.assertListEqual(['a','b'], list(bundle.test.columns))
        rs = caching_extractor.extract(ibundle.change_index(ibundle.index_frame.iloc[:3]))
        self.assertListEqual([0,1,2], list(rs.a))
        self.assertListEqual([0, 10, 20], list(rs.b))

        tdf = Query.en(range(4)).select(lambda z: dict(a=2*z, b=3*z)).to_dataframe()
        test_bundle = DataBundle(index=tdf)
        ibundle = IndexedDataBundle(test_bundle.index, test_bundle)
        caching_extractor.preprocess_bundle(ibundle)
        self.assertListEqual(['index','test'], list(test_bundle.data_frames))
        trs = caching_extractor.extract(ibundle.change_index(ibundle.index_frame.iloc[:3]))
        self.assertListEqual([0,2,4], list(trs.a))
        self.assertListEqual([0, 3, 6], list(trs.b))


    def test_batch_size(self):
        df = Query.en(range(10)).select(lambda z: dict(a=z, b=10 * z, c=100 * z)).to_dataframe()
        bundle = DataBundle(index = df)
        inner_extractor = PlainExtractor.build('t').index().apply(take_columns=['a', 'b'])
        caching_extractor = PrecomputingExtractor('test', inner_extractor, index_size_for_precomputing=4)
        ibundle = IndexedDataBundle(bundle.index, bundle)
        caching_extractor.fit(ibundle)
        dfs = caching_extractor._precompute(ibundle)
        self.assertEqual(3, len(dfs))
        self.assertEqual(4, dfs[0].shape[0])
        self.assertEqual(4, dfs[1].shape[0])
        self.assertEqual(2, dfs[2].shape[0])
        self.assertEqual([80, 90], list(dfs[2].b))

    def test_batch_filter(self):
        df = Query.en(range(10)).select(lambda z: dict(a=z, b=10 * z, c=100 * z)).to_dataframe()
        bundle = DataBundle(index=df)
        inner_extractor = PlainExtractor.build('t').index().apply(take_columns=['a', 'b'])
        caching_extractor = PrecomputingExtractor('test', inner_extractor, index_size_for_precomputing=3, index_filter_for_precomputing=lambda z: z.loc[z.a % 2 == 0].index)
        caching_extractor.fit(IndexedDataBundle(bundle.index, bundle))
        dfs = caching_extractor._precompute(IndexedDataBundle(bundle.index,bundle))
        print(dfs)
        self.assertEqual(2, len(dfs))
        self.assertEqual(3, dfs[0].shape[0])
        self.assertEqual(2, dfs[1].shape[0])
        self.assertListEqual([60, 80], list(dfs[1].b))








