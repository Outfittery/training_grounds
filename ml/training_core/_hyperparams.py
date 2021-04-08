from typing import *

from datetime import timedelta



def _type_value(type: str, value: str):
    if type == 'str':
        return value
    elif type == 'int':
        return int(value)
    elif type == 'float':
        return float(value)
    elif type == 'bool':
        return bool(value)
    elif type == 'timedelta-day':
        return timedelta(days=float(value))
    raise ValueError(f'Unknown type `{type}` for value `{value}`')


def _get(path_part, object):
    if isinstance(object, list):
        return object[int(path_part)]
    elif isinstance(object, dict):
        return object[path_part]
    else:
        return getattr(object, path_part)


def _set(path_part, object, value):
    if isinstance(object, list):
        index = int(path_part)
        if index >= len(object):
            raise ValueError(f'Non-existing index {path_part} in list when setting the parameter')
        object[index] = value
    elif isinstance(object, dict):
        object[path_part] = value
    else:
        try:
            getattr(object, path_part)
        except:
            raise ValueError(f'Non existing field {path_part} in object when setting the parameter')
        setattr(object, path_part, value)


def _apply_hyperparam(key: str, value: str, object: Any):
    parts = key.split(':')
    if len(parts) == 1:
        path, type = parts[0], 'str'
    elif len(parts) == 2:
        path, type = parts
    else:
        raise ValueError('Parameter key {0} must contain no more than one semicolon `:`. Key was {0}'.format(key))

    value = _type_value(type, value)

    path_parts = path.split('.')
    for index, path_part in enumerate(path_parts):
        if index != len(path_parts) - 1:
            object = _get(path_part, object)
        else:
            _set(path_part, object, value)


def _apply_hyperparams(params: Dict[str, str], object: Any):
    """
    Applies the hyperparameters values to the given object
    """
    for key, value in params.items():
        _apply_hyperparam(key, value, object)
