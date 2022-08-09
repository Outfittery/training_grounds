from typing import *
from .artificiers import Artificier, ArtificierArguments
from .metrics import Metric
from yo_fluq_ds import fluq
import pandas as pd

class MulticlassWinningArtificier(Artificier):
    def __init__(self,
                 order_cap=None,
                 add_winner=True
                 ):
        self.order_cap = order_cap
        self.add_winner = add_winner

    def _unstack(self, df, prefix, value_column_name):
        columns = [c for c in df.columns if c.startswith(prefix)]
        pdf = df[columns].unstack().reset_index()
        pdf.columns = ['label', 'idx', value_column_name]
        pdf.label = pdf.label.str.replace(prefix, '')
        return pdf

    def _pivot(self, pdf, column):
        if self.order_cap is not None:
            pdf = pdf.loc[pdf.ord < self.order_cap]
        rdf = pdf.pivot_table(index='idx', columns='ord', values=column, aggfunc=lambda z: z)
        rdf.columns = [f'ord_{i}_{column}' for i in rdf.columns]
        return rdf

    def run_before_metrics(self, args: ArtificierArguments):
        df = args.result.result_df
        pdf = self._unstack(df, 'predicted_', 'predicted_score')
        pdf = pdf.feed(fluq.add_ordering_column('idx', ('predicted_score', False), 'ord'))

        tdf = self._unstack(df, 'true_', 'true_score')
        pdf = pdf.merge(tdf.set_index(['idx', 'label']), left_on=['idx', 'label'], right_index=True, how='left')

        rdf = self._pivot(pdf, 'label')
        rdf = rdf.merge(self._pivot(pdf, 'predicted_score'), left_index=True, right_index=True)

        if self.add_winner:
            pdf['winner'] = pdf.true_score > 0.5
            wdf = pdf.loc[pdf.winner]
            s = wdf.groupby('idx').size()
            if (s>1).any():
                raise ValueError(f'Multiple winners for samples: {list(s.loc[s>1].index)}')
            rdf = rdf.merge(wdf.set_index('idx').ord.to_frame('win_result_at'), left_index=True, right_index=True)
            rdf = rdf.merge(wdf.set_index('idx').label.to_frame('win_true_label'), left_index=True, right_index=True)

        for c in rdf.columns:
            df[c] = rdf[c]


class RecallAtK(Metric):
    def __init__(self, k):
        self.k = k

    def get_names(self) -> List[str]:
        return [f'recall_at_{self.k}']

    def measure(self, result_df: pd.DataFrame, source_data: Any) -> List[Any]:
        return [(result_df.win_result_at<self.k).sum()/result_df.shape[0]]

