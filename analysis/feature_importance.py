from typing import *

import pandas as pd

from sklearn.metrics import roc_auc_score, mean_absolute_error
from functools import partial

from ..ml import single_frame_training as sft, dft
from sklearn.linear_model import Ridge, LogisticRegression

class FeatureSignificance:
    class Artificier(sft.Artificier):
        def run_before_metrics(self, model_info: sft.ArtificierArguments):
            column_names_keeper = model_info.result.model[2]  # type: sft.ColumnNamesKeeper
            column_names = column_names_keeper.column_names_
            model = model_info.result.model[3]
            coeficients = model.coef_
            if isinstance(model, LogisticRegression):
                model_info.result.significance = pd.Series(coeficients[0], index=column_names)
            elif isinstance(model, Ridge):
                model_info.result.significance = pd.Series(coeficients, index=column_names)
            else:
                raise ValueError(f'Unexpected model type {type(model)}')

    @staticmethod
    def run_task(
            model_class,
            evaluation,
            metric_pool,
            df: pd.DataFrame,
            features: List[str],
            label: str,
            folds_count: Optional[int],
            model_kwargs
    ):
        if folds_count is None:
            splitter = sft.IdentitySplitter()
        else:
            splitter = sft.FoldSplitter(fold_count=folds_count)

        task = sft.SingleFrameTrainingTask(
            data_loader=sft.DataFrameLoader(label),
            model_provider=sft.ModelProvider(
                constructor=sft.ModelConstructor(model_class, **model_kwargs),
                transformer=dft.DataFrameTransformerFactory.default_factory(features)
            ),
            evaluator=evaluation,
            splitter=splitter,
            metrics_pool=metric_pool,
            artificers=[
                FeatureSignificance.Artificier()
            ]
        )

        result = task.run(df)
        sdf = pd.DataFrame([run['significance'] for run in result['runs'].values()])
        return sdf, result, task





    @staticmethod
    def for_classification_task(
            df: pd.DataFrame,
            features: List[str],
            label: str,
            folds_count: Optional[int],
            model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = dict(penalty='l2')

        return FeatureSignificance.run_task(
            'sklearn.linear_model:LogisticRegression',
            sft.Evaluation.binary_classification,
            sft.MetricPool().add_sklearn(roc_auc_score),
            df,
            features,
            label,
            folds_count,
            model_kwargs
        )

    @staticmethod
    def for_regression_task(
            df: pd.DataFrame,
            features: List[str],
            label: str,
            folds_count: Optional[int],
            model_kwargs=None
    ):
        return FeatureSignificance.run_task(
            'sklearn.linear_model:Ridge',
            sft.Evaluation.regression,
            sft.MetricPool().add_sklearn(mean_absolute_error),
            df,
            features,
            label,
            folds_count,
            {}
        )
