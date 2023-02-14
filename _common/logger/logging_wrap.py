from typing import *

import logging
import simplejson
import sys
import datetime
import traceback
import platform

from logging import getLogger, StreamHandler, Formatter, LogRecord
from dateutil import tz

from .logger_interface import LoggerInterface


class FieldsCompatibleFormatter(Formatter):
    def __init__(self,
                 formatting_method: Callable[[Dict], str],
                 ):
        super(FieldsCompatibleFormatter, self).__init__()
        self.fields = {}  # type: Dict
        self.temp_fields = {}  # type: Dict
        self.formatting_method = formatting_method

    def set_fields(self, fields):
        self.fields = fields

    def format(self, record: LogRecord) -> str:
        data = self.collect_data(record, self.fields)
        s = self.formatting_method(data)
        return s

    def collect_data(self, record: LogRecord, fields: dict):
        data = {
            '@timestamp': datetime.datetime.now(),
            'message': record.getMessage(),
            'levelname': record.levelname,
            'logger': record.name,
            'path': record.pathname,
            'path_line': record.lineno
        }

        ex_info = sys.exc_info()
        if ex_info is not None and ex_info[0] is not None:
            data['exception_type'] = str(ex_info[0])
            data['exception_value'] = str(ex_info[1])
            data['exception_details'] = ''.join(traceback.format_exception(*ex_info))

        data.update(fields)
        return data


class TGStdoutHandler(StreamHandler):
    def __init__(self,
                 formatter: FieldsCompatibleFormatter
                 ):
        StreamHandler.__init__(self, stream=sys.stdout)
        self.setFormatter(formatter)


class LoggingWrap(LoggerInterface):
    def __init__(self,
                 name: str,
                 level: int,
                 formatting_method: Callable[[Dict], str],
                 stack_shift: Optional[int] = 4
                 ):
        self._logger = getLogger(name)
        self._logger.setLevel(level)
        self._formatter = FieldsCompatibleFormatter(formatting_method)
        self._handler = TGStdoutHandler(self._formatter)
        self._logger.addHandler(self._handler)
        self.stack_shift = stack_shift
        version = tuple(platform.python_version().split('.'))
        if version <= ('3', '8', '0'):
            self.stack_shift = None

    def set_extra_fields(self, fields: Dict[str, Dict]):
        self._formatter.set_fields(fields)

    def output(self, method, object):
        if self.stack_shift is not None:
            getattr(self._logger, method)(object, stacklevel=self.stack_shift)
        else:
            getattr(self._logger, method)(object)

    def close(self):
        self._logger.handlers.remove(self._handler)
