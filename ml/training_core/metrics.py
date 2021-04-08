from typing import *
import pandas as pd
from .arch import Artificier, ArtificierArguments



class Metric:
    def get_names(self) -> List[str]:
        raise NotImplementedError()

    def measure(self, result_df: pd.DataFrame, source_data: Any) -> List[Any]:
        raise NotImplementedError()



class SklearnMetric(Metric):
    def __init__(self, method, **kwargs):
        self._name = method.__name__
        self._kwargs = kwargs
        self._method = method

    def get_names(self) -> List[str]:
        return [self._name]

    def measure(self, result_df: pd.DataFrame, source_data: Any) -> List[Any]:
        return [self._method(result_df.true, result_df.predicted, **self._kwargs)]


class MetricPool(Artificier):
    """
    A class representing a collection of metrics.
    Implements FluentAPI, add metrics one by one
    """
    def __init__(self):
        self.metrics = []  # type: List[Metric]

    def add_sklearn(self, method, **kwargs) -> 'MetricPool':
        """
        Adds one metric to the provider
        Args:
            method: the method to call
            kwargs: additional arguments to the method (the first two are true and predicted values)

        Returns: self

        """
        return self.add(SklearnMetric(method, **kwargs))


    def add(self, metric: Metric) -> 'MetricPool':
        self.metrics.append(metric)
        return self

    def get_metrics_names(self) -> List[str]:
        return [n for m in self.metrics for n in m.get_names()]

    def run(self, args: ArtificierArguments):
        """
        Computes the metrics for a given dataframe

        Returns: dictionary, keys are ``[metric_name]_[stage]``, values are the output of the metrics' methods
        """
        df = args.result.result_df
        result_metrics = {}
        for stage in df.stage.unique():
            for metric in self.metrics:
                tdf = df.loc[df.stage == stage]
                result = metric.measure(tdf, args.source_data)
                names = metric.get_names()
                for key, value in zip(names,result):
                    result_metrics[key + '_' + str(stage)] = value
        args.result.metrics=result_metrics
