from typing import *

import numpy as np

from .combinators import Pipeline



def _flatten_dict_rec(result, current_key, d):
    if isinstance(d, dict):
        if current_key != '':
            current_key += '_'
        for key, value in d.items():
            _flatten_dict_rec(result, current_key + str(key), value)
    else:
        result[current_key] = d


def flatten_dict(d: Dict) -> Dict:
    """
    Converts hierarchy of dictionaries into one flattened dictionary. The keys are joined with '_'
    """
    result = {}
    _flatten_dict_rec(result, '', d)
    return result


def np_bool_to_bool(d: Dict) -> Dict:
    """
    In the dictionary, replace all values of type :class:`np.bool` with :class:`bool`.
    Updates the dictionary and also returns it, so it's not a pure function.
    """
    for key in d.keys():
        if isinstance(d[key], np.bool_):
            d[key] = bool(d[key])
    return d


def default_tail_pipeline():
    return Pipeline(
        flatten_dict,
        np_bool_to_bool
    )
