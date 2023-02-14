from typing import *

import json
import logging

from .logging_wrap import LoggingWrap
import datetime

PRIMITIVES = (int, bool, str, float)


class KibanaTypeProcessor:
    def __init__(self):
        self.field_types = {}

    def _observe_types(self, d: Dict):
        for key, value in d.items():
            if key in self.field_types:
                continue
            if value is None:
                continue
            if type(value) in PRIMITIVES:
                self.field_types[key] = type(value).__name__
            elif isinstance(value, list) or isinstance(value, dict):
                self.field_types[key] = 'json'
            else:
                self.field_types[key] = 'other'

    def _replace_non_primitives(self, d: Dict):
        result = {}
        for key, value in d.items():
            if value is None:
                result[key] = value
            elif type(value) in PRIMITIVES:
                result[key] = value
            elif isinstance(value, list) or isinstance(value, dict):
                result[key] = json.dumps(value)
            elif isinstance(value, datetime.datetime):
                result[key] = value.astimezone(datetime.timezone.utc).isoformat()
            else:
                result[key] = str(value)
        return result

    def _default_value(self, t):
        if t == 'int':
            return -1
        elif t == 'float':
            return 0.0
        elif t == 'bool':
            return False
        elif t == 'str':
            return ''
        elif t == 'json':
            return 'null'
        elif t == 'other':
            return ''
        else:
            raise ValueError(f'Wrong type to call _default_value with: {t}')

    def _replace_nones(self, d: Dict):
        result = {}
        for key, value in d.items():
            if value is None:
                if key in self.field_types:
                    result[key] = self._default_value(self.field_types[key])
            else:
                result[key] = value
        return result

    def process(self, d: Dict):
        self._observe_types(d)
        d = self._replace_non_primitives(d)
        d = self._replace_nones(d)
        return d


class KibanaLoggingWrap(LoggingWrap):
    def __init__(self, scope='tg', level=logging.INFO):
        self.type_processor = KibanaTypeProcessor()
        super(KibanaLoggingWrap, self).__init__(scope, level, self._serialize)

    def _serialize(self, d):
        d = self.type_processor.process(d)
        return json.dumps(d)
