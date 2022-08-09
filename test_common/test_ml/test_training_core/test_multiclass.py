from unittest import TestCase
from tg.common.ml.training_core import MulticlassWinningArtificier, ArtificierArguments, TrainingResult
import pandas as pd

pd.options.display.width = None

df = pd.DataFrame(dict(
    true_A = [0, 0, 0, 0, 1, 1],
    true_B = [0, 0, 1, 1, 0, 0],
    true_C = [1, 1, 0, 0, 0, 0],
    predicted_A = [3, 1, 1, 2, 2, 3],
    predicted_B = [2, 2, 2, 3, 1, 2],
    predicted_C = [1, 3, 3, 1, 3, 1]
))

def run(*args):
    res = TrainingResult()
    res.result_df = df.copy()
    art = MulticlassWinningArtificier(*args)
    args = ArtificierArguments(res, None)
    art.run_before_metrics(args)
    rdf = args.result.result_df
    return rdf

class MulticlassTestCase(TestCase):
    def test_artificier(self):
        rdf = run(None, False)
        for i in range(3):
            for label in ['label','predicted_score']:
                self.assertIn(f'ord_{i}_{label}', rdf.columns)
            self.assertListEqual([3-i]*6, list(rdf[f'ord_{i}_predicted_score']))
        self.assertListEqual(['A','C','C','B','C','A'], list(rdf.ord_0_label))
        self.assertListEqual(['B', 'B', 'B', 'A', 'A', 'B'], list(rdf.ord_1_label))
        self.assertListEqual(['C', 'A', 'A', 'C', 'B', 'C'], list(rdf.ord_1_label))

    def test_artificier_winner(self):
        rdf = run(None, True)
        self.assertListEqual(['C','C','B','B','A','A'], list(rdf.win_true_label))
        self.assertListEqual([2,0,1,0,1,0], list(rdf.win_result_at))

    def test_artificier_cap(self):
        rdf = run(2, False)
        for i in range(3):
            for label in ['label', 'predicted_score']:
                if i<2:
                    self.assertIn(f'ord_{i}_{label}', rdf.columns)
                else:
                    self.assertNotIn(f'ord_{i}_{label}', rdf.columns)


