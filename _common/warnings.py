from typing import *

import copy



class OldTGWarning:
    def __init__(self, message, **kwargs):
        self.message = message
        self.payload = kwargs
        self.full_message = self.message.format(**self.payload)

    def __repr__(self):
        return self.full_message


class TGWarning:
    def __init__(self,
                 message,
                 *levels: Dict[str, Any]
                 ):
        self.levels = levels
        self.message = message

    def __repr__(self):
        result = self.message + ' ' + ' '.join(
            str(key) + '=' + str(value) for level in self.levels for key, value in level.items())
        return result


class TGWarningStorageLevel:
    def __init__(self):
        self.keys = None  # type: Optional[List]
        self.values = None  # type: Optional[Dict]
        self.count = 0

    def add_recursive(self, warning, index):
        self.count += 1
        if index < len(warning.levels):
            top = warning.levels[index]
            if self.keys is None:
                self.keys = list(top.keys())
                self.values = dict()
            else:
                if len(self.keys) != len(top):
                    raise ValueError(
                        f"Inconsistent keys for message {warning.message}, stored keys {self.keys}, adding keys {list(top.keys())}")
            values = tuple(top[key] for key in self.keys)
            if values not in self.values:
                self.values[values] = TGWarningStorageLevel()
            self.values[values].add_recursive(warning, index + 1)

    def get_report_iter(self, record, index, max_index):
        if (max_index is not None and index >= max_index) or self.keys is None:
            record['_count'] = self.count
            yield record
        else:
            for values, level in self.values.items():
                rec = copy.copy(record)
                for key, value in zip(self.keys, values):
                    rec[key] = value
                for r in level.get_report_iter(rec, index + 1, max_index):
                    yield r


class TGWarningStorageClass:
    def __init__(self):
        self.storages = dict()  # type: Dict[str, TGWarningStorageLevel]

    def add_warning(self, message: str, *levels: Dict[str, Any]):
        warning = TGWarning(message, *levels)
        if warning.message not in self.storages:
            self.storages[warning.message] = TGWarningStorageLevel()
        self.storages[warning.message].add_recursive(warning, 0)

    def _get_report_iter(self, max_level):
        for storage, level in self.storages.items():
            for r in level.get_report_iter({'_message': storage}, 0, max_level):
                yield r

    def get_report(self, max_level=None):
        return list(self._get_report_iter(max_level))

    def clear(self):
        self.storages = dict()


TGWarningStorage = TGWarningStorageClass()
