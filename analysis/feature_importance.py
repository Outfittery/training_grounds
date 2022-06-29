from typing import *

import pandas as pd

from sklearn.metrics import roc_auc_score
from functools import partial

from ..ml import single_frame_training as sft, dft


class FeatureSignificance:
    class Artificier(sft.Artificier):
        def run(self, model_info):
            column_names_keeper = model_info.result.model[2]  # type: sft.ColumnNamesKeeper
            column_names = column_names_keeper.column_names_
            coeficients = model_info.result.model[3].coef_
            model_info.result.significance = pd.Series(coeficients[0], index=column_names)

    @staticmethod
    def for_classification_task(
            df: pd.DataFrame,
            features: List[str],
            label: str,
            folds_count,
            model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = dict(penalty='l2')

        task = sft.SingleFrameTrainingTask(
            data_loader=sft.DataFrameLoader(label),
            model_provider=sft.ModelProvider(
                constructor=sft.ModelConstructor('sklearn.linear_model:LogisticRegression', **model_kwargs),
                transformer=dft.DataFrameTransformerFactory.default_factory(features)
            ),
            evaluator=sft.Evaluation.binary_classification,
            splitter=sft.FoldSplitter(fold_count=folds_count),
            metrics_pool=sft.MetricPool().add_sklearn(roc_auc_score),
            artificers=[
                FeatureSignificance.Artificier()
            ]
        )

        result = task.run(df)
        sdf = pd.DataFrame([run['significance'] for run in result['runs'].values()])
        return sdf
