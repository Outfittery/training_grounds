from unittest import TestCase
from tg.common.ml.miscellaneous.glove import GloveProcessor
from tg.common import Loc
from yo_fluq_ds import FileIO
import pandas as pd

scores_text = '''
a 1 0 1
b 0 1 1
c 1 1 1
<unk> 1 1 0
'''

def get_scores():
    fname = Loc.temp_path/'glove_scores'
    FileIO.write_text(scores_text.strip(), fname)
    return GloveProcessor.read_glove_scores(fname)

def s(args):
    return pd.Series(list(args), index=[f'i{i}' for i in range(len(args))])


class GloveScoresApplicationTestCase(TestCase):
    def test_simple(self):
        df = get_scores()
        result = GloveProcessor.apply_scores(s('abc'), s('bca'), df)
        self.assertListEqual(['i0', 'i1', 'i2'], list(result.index))
        self.assertListEqual([1,2,2], list(result))

    def test_missing(self):
        df = get_scores()
        result = GloveProcessor.apply_scores(s('aad'), s('add'), df)
        self.assertListEqual([False,True,True], list(result.isnull()))

    def test_unk(self):
        df = get_scores()
        result = GloveProcessor.apply_scores(s('ccaad'), s('cbadd'), df, '<unk>')
        self.assertListEqual([3,2,2,1,2], list(result))