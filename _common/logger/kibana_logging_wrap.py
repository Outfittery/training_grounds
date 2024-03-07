from typing import *

import json
import logging

from .logging_wrap import LoggingWrap
import datetime

PRIMITIVES = (int, bool, str, float)


class KibanaTypeProcessor:
    def __init__(self):
        self.field_types = {}

    def _observe_types(self, d: Dict, parent: str):
        for key, value in d.items():
            key = parent + str(key)
            if key in self.field_types:
                continue
            if value is None:
                continue
            if type(value) in PRIMITIVES + (list, dict):
                self.field_types[key] = type(value).__name__
            else:
                self.field_types[key] = 'other'

    def _replace_non_primitives(self, d: Dict, parent: str):
        result = {}
        for key, value in d.items():
            if value is None:
                result[key] = value
            elif type(value) in PRIMITIVES:
                result[key] = value
            elif isinstance(value, list):
                result[key] = [json.dumps(e) for e in value]
            elif isinstance(value, dict):
                result[key] = self.process(value, parent=parent + str(key))
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
        elif t == 'dict':
            return {}
        elif t in ['str', 'list', 'other']:
            return ''
        else:
            raise ValueError(f'Wrong type to call _default_value with: {t}')

    def _replace_empty(self, d: Dict, parent: str):
        result = {}
        for key, value in d.items():
            if value in [None, []]:
                if parent + key in self.field_types:
                    result[key] = self._default_value(self.field_types[parent + key])
            else:
                result[key] = value
        return result

    def process(self, d: Dict, parent: str = ''):
        self._observe_types(d, parent)
        d = self._replace_non_primitives(d, parent)
        d = self._replace_empty(d, parent)
        return d


class KibanaLoggingWrap(LoggingWrap):
    def __init__(self, scope='tg', level=logging.INFO, human_readable=False):
        self.type_processor = KibanaTypeProcessor()
        self.human_readable = human_readable
        super(KibanaLoggingWrap, self).__init__(scope, level, self._serialize)

    def _serialize(self, d):
        d = self.type_processor.process(d)
        if self.human_readable:
            return json.dumps(d, indent=4)
        else:
            return json.dumps(d)
