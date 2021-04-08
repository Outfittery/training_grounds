from unittest import TestCase
import pandas as pd
from tg.common.ml.batched_training import Extractor, DataBundle, DirectExtractor, LeftJoinExtractor, CombinedExtractor, IndexedDataBundle, DataBundle

class ExtractorsTestCase(TestCase):
    def make_df(self, extractor: Extractor, bundle: IndexedDataBundle):
        extractor.fit(bundle.bundle)
        index = bundle.bundle.index_frame
        if bundle.index is not None:
            index = index.loc[bundle.index]
        return extractor.extract(index, bundle.bundle)


    def test_direct(self):
        index = pd.DataFrame([
            ("c", 3),
            ("a", 6),
            ("b", 3),
            ('d', -1)
        ],
            columns=['ind','a']
        ).set_index('ind')

        data = pd.DataFrame([
            (3,10),
            (6,20),
            (5,30)],
            columns=['ind','x']
        ).set_index('ind')

        bundle = DataBundle(index_frame=index, data_frames=dict(a=data)).as_indexed()

        self.assertRaises(ValueError, lambda: self.make_df(DirectExtractor('a'), bundle))

        bundle.index=['c','a','b']

        res = self.make_df(DirectExtractor('a'), bundle)
        self.assertEqual('a',res.key)
        self.assertListEqual(['c','a','b'], list(res.value.index))
        self.assertListEqual([10,20,10],list(res.value.x))
        self.assertListEqual(['x'],list(res.value.columns))

        bundle = DataBundle(index_frame=index, data_frames=dict(test=data)).as_indexed(['c','a','b'])

        res = self.make_df(DirectExtractor('a', custom_dataframe_name='test'), bundle)
        self.assertEqual('a', res.key)
        self.assertListEqual(['c', 'a', 'b'], list(res.value.index))
        self.assertListEqual([10, 20, 10], list(res.value.x))
        self.assertListEqual(['x'], list(res.value.columns))


    def test_left_join_wrong_index(self):
        index = pd.DataFrame(dict(a=[2,3]))
        data = pd.DataFrame(dict(ind=[2,2,3], x=[4,5,6])).set_index('ind')
        bundle = DataBundle(index,data_frames=dict(frame=data)).as_indexed()
        self.assertRaises(ValueError, lambda: self.make_df(LeftJoinExtractor('tst',['a'],'frame'),bundle))


    def test_left_join_two_columns(self):
        index = pd.DataFrame(dict(
            order=[1,1,2,2],
            month=[1,2,4,5],
        ))
        data = pd.DataFrame([
            (1,1,100),
            (1,2,101),
            (1,3,102),
            (2,3,103),
            (2,4,104),
            (2,5,105)
        ], columns=['order','month','val']).set_index(['order','month'])
        bundle = DataBundle(index,data_frames=dict(test=data)).as_indexed()
        res = self.make_df(LeftJoinExtractor('a',['order','month'],'test'),bundle)
        self.assertEqual('a',res.key)
        self.assertListEqual(['order','month','val'], list(res.value.columns))
        self.assertListEqual([100,101,104,105],list(res.value.val))


    def test_combined_extractors(self):
        index = pd.DataFrame(dict(
            order=[1, 1, 2, 2],
            month=[1, 2, 4, 5],
        ))
        data = pd.DataFrame([
            (1, 1, 100),
            (1, 2, 101),
            (1, 3, 102),
            (2, 3, 103),
            (2, 4, 104),
            (2, 5, 105)
        ], columns=['order', 'month', 'val']).set_index(['order', 'month'])
        order_data = pd.DataFrame([
            (1, 'b'),
            (2, 'c')
        ], columns = ['ind','letter']).set_index('ind')
        bundle = DataBundle(index, dict(order=order_data, order_month = data)).as_indexed()

        ext = CombinedExtractor(
            'oc',
            [
                DirectExtractor('order'),
                LeftJoinExtractor(
                    'comb',
                    ['order','month'],
                    'order_month')
            ]
        )

        rs = self.make_df(ext, bundle)
        self.assertListEqual(['letter','order','month','val'], list(rs.value.columns))
        self.assertListEqual(['b','b','c','c'], list(rs.value.letter))
        self.assertListEqual([100,101,104,105], list(rs.value.val))









