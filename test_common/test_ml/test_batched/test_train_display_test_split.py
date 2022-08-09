from tg.common.ml.batched_training import train_display_test_split
from unittest import TestCase
import pandas as pd
from yo_fluq_ds import *

class TestDisplaySplitTestCase(TestCase):
    def test_split(self):
        df = pd.DataFrame(dict(a=['A']*10000+['B']*100))
        df['split'] = train_display_test_split(df, 0.3, 0.2, 'a')
        frc = df.groupby('split').feed(fluq.fractions())
        self.assertAlmostEqual(frc['display'], 0.2, 2)
        self.assertAlmostEqual(frc['test'], 0.3, 2)
        self.assertAlmostEqual(frc['train'], 0.5, 2)


        frc = df.loc[df.split=='test'].groupby('a').feed(fluq.fractions())
        self.assertAlmostEqual(frc['A'],0.99, 2)



