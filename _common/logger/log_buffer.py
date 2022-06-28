import datetime
import json
import sys
import logging
from .logger_root import Logger
from yo_fluq_ds import Query
from .logging_wrap import LoggingWrap
from io import StringIO

class LogBuffer:
    def _serialize(self, data):
        for key, value in list(data.items()):
            if isinstance(value, datetime.datetime):
                data[key] = str(value)
        return json.dumps(data)


    def __init__(self, custom_wrap_factory=None, **keys):
        self.buffer = StringIO()
        sys.stdout = self.buffer
        if custom_wrap_factory is None:
            wrap = LoggingWrap(
                'tg',
                logging.INFO,
                self._serialize
            )
            Logger.reset(wrap, keys)
        else:
            Logger.reset(custom_wrap_factory(), keys)

    def parse(self):
        val = self.buffer.getvalue()
        lines = [z for z in val.split('\n') if z.strip() != '']
        objects = [json.loads(z) for z in lines]
        return Query.en(objects)

    def transpose(self, *columns):
        ps = self.parse().to_list()
        return {c: [z.get(c,'#') for z in ps] for c in columns}

    def read(self):
        val = self.buffer.getvalue()
        lines = [z for z in val.split('\n') if z.strip() != '']
        return Query.en(lines)