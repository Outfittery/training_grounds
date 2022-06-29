import logging

from .logging_wrap import LoggingWrap


def _debug_formatting(data):
    return f"{data['@timestamp']} {data['levelname']}: {data['message']}"


class DebugLoggingWrap(LoggingWrap):
    def __init__(self, scope='tg', level=logging.DEBUG):
        super(DebugLoggingWrap, self).__init__(
            scope,
            level,
            _debug_formatting
        )
