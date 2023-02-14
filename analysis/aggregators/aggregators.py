import numpy as np

from functools import partial
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm

from .architecture import *


def _cname(column, suffix, multilevel):
    if multilevel:
        return (column, suffix)
    else:
        return f'{column}_{suffix}'


class Aggregators:
    @staticmethod
    def _proportion_aggregator(s, row, pValue, method, multilevel):
        values = s.unique()
        for v in values:
            if v != True and v != False:
                raise ValueError(f'Aggregation series {s.name}, value {v} was found. Expected only True and False')
        cf = proportion_confint(s.sum(), s.shape[0], 1 - pValue, method)
        row[_cname(s.name, 'lower', multilevel)] = cf[0]
        row[_cname(s.name, 'upper', multilevel)] = cf[1]
        row[_cname(s.name, 'value', multilevel)] = (cf[1] + cf[0]) / 2
        row[_cname(s.name, 'error', multilevel)] = (cf[1] - cf[0]) / 2

    @staticmethod
    def _percentile_confint(s, row, pValue, multilevel):
        lower = np.percentile(s, 100 * (1 - pValue) / 2)
        upper = np.percentile(s, 100 - 100 * (1 - pValue) / 2)
        row[_cname(s.name, 'lower', multilevel)] = lower
        row[_cname(s.name, 'upper', multilevel)] = upper
        row[_cname(s.name, 'value', multilevel)] = (upper + lower) / 2
        row[_cname(s.name, 'error', multilevel)] = (upper - lower) / 2

    @staticmethod
    def _normal_confint(s, row, pValue, multilevel):
        loc = s.mean()
        scale = s.std()
        cf = norm.interval(0.95, loc=loc, scale=scale)
        row[_cname(s.name, 'lower', multilevel)] = cf[0]
        row[_cname(s.name, 'upper', multilevel)] = cf[1]
        row[_cname(s.name, 'value', multilevel)] = (cf[1] + cf[0]) / 2
        row[_cname(s.name, 'error', multilevel)] = (cf[1] - cf[0]) / 2

    @staticmethod
    def proportion_confint(columns=None, pValue=0.95, method='normal', multilevel_column=False):
        return CustomAggregator(columns, partial(Aggregators._proportion_aggregator, pValue=pValue, method=method, multilevel=multilevel_column))

    @staticmethod
    def percentile_confint(columns=None, pValue=0.95, multilevel_column=False):
        return CustomAggregator(columns, partial(Aggregators._percentile_confint, pValue=pValue, multilevel=multilevel_column))

    @staticmethod
    def normal_confint(columns=None, pValue=0.95, multilevel_column=False):
        return CustomAggregator(columns, partial(Aggregators._normal_confint, pValue=pValue, multilevel=multilevel_column))

    @staticmethod
    def combine(*aggs):
        return CombinedAggregator(aggs)

    @staticmethod
    def pandas(**kwargs):
        return PandasAggregator(**kwargs)
