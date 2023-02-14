from unittest import TestCase
from tg.common.ml.training_core import MulticlassMetrics
import pandas as pd

pd.options.display.width = None

df = pd.DataFrame(dict(
    true_label_A = [0, 0, 0, 0, 1, 1],
    true_label_B = [0, 0, 1, 1, 0, 0],
    true_label_C = [1, 1, 0, 0, 0, 0],
    predicted_label_A = [3, 1, 1, 2, 2, 3],
    predicted_label_B = [2, 2, 2, 3, 1, 2],
    predicted_label_C = [1, 3, 3, 1, 3, 1]
))
for c in df.columns:
    if c.startswith('predicted'):
        df[c] = df[c]/10
df.index=df.index*10

class MulticlassMetricsTestCase(TestCase):
    def test_unwrapping(self):
        xdf = MulticlassMetrics.get_winner_and_rating(df)
        s = [i*10 for i in range(6) for _ in range(3) ]
        self.assertEqual(s, list(xdf['sample']))
        self.assertEqual([0,1,2]*6, list(xdf.prediction_rating))
        self.assertEqual([0.3,0.2,0.1]*6, list(xdf.predicted))
        self.assertEqual(['A','B','C','C','B','A','C','B','A','B','A','C','C','A','B','A','B','C'], list(xdf.label))
        self.assertEqual(['C','C','B','B','A','A'], list(xdf.loc[xdf.true>0.5].label))

    def test_metrics(self):
        xdf = MulticlassMetrics.get_winner_and_rating(df)
        print(xdf)
        m = MulticlassMetrics(True, True, [1,2])
        r = {k:v for k, v in zip(m.get_names(), m.measure(df, None))}
        self.assertEqual(0.5, r['accuracy'])
        self.assertEqual(4/6, r['rating'])
        self.assertEqual(0.5, r['recall_at_1'])
        self.assertEqual(5/6, r['recall_at_2'])



