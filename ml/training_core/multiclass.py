from typing import *
from ..training_core import Metric
from yo_fluq_ds import fluq

class MulticlassMetrics(Metric):
    def __init__(self, add_accuracy=True, add_rating=False, recall_at: Union[None, int, Iterable[int]] = None):
        self.add_accuracy = add_accuracy
        self.add_rating = add_rating
        if recall_at is None:
            self.recall_at = []
        elif isinstance(recall_at, int):
            self.recall_at = [recall_at]
        else:
            self.recall_at = list(recall_at)

    def get_names(self):
        result = []
        if self.add_accuracy:
            result.append('accuracy')
        if self.add_rating:
            result.append('rating')
        for i in self.recall_at:
            result.append(f'recall_at_{i}')
        return result

    @staticmethod
    def get_winner_and_rating(df, label_prefix = 'true_label_'):
        labels = []
        for c in df.columns:
            if c.startswith(label_prefix):
                labels.append(c.replace(label_prefix, ''))

        def ustack(df, prefix, cols, name):
            df = df[[prefix + c for c in cols]]
            df.columns = [c for c in cols]
            df = df.unstack().to_frame(name)
            return df

        predicted = ustack(df, 'predicted_label_', labels, 'predicted')
        true = ustack(df, 'true_label_', labels, 'true')
        df = predicted.merge(true, left_index=True, right_index=True).reset_index()
        df.columns = ['label', 'sample', 'predicted', 'true']
        df = df.feed(fluq.add_ordering_column('sample', ('predicted', False), 'prediction_rating'))
        df = df.sort_values(['sample','prediction_rating'])
        df = df.reset_index(drop=True)
        return df

    def measure(self, df, _):
        df = MulticlassMetrics.get_winner_and_rating(df)
        match = (df.loc[df.prediction_rating == 0].set_index('sample').true > 0.5)
        rating = df.loc[df.true > 0.5].set_index('sample').prediction_rating
        result = []
        if self.add_accuracy:
            result.append(match.mean())
        if self.add_rating:
            result.append(rating.mean())
        for i in self.recall_at:
            r = (df.loc[df.true>0.5].prediction_rating<i).mean()
            result.append(r)
        return result