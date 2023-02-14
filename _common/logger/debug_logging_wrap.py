import logging

from .logging_wrap import LoggingWrap


def _debug_formatting(data):
    return f"{data['@timestamp']} {data['levelname']}: {data['message']}"


def _debug_formatting_with_keys(data):
    other_keys = ' | '.join([f'{k} {v}' for k,v in data.items() if k not in ['@timestamp','levelname','message']])
    msg = f"{data['@timestamp']} {data['levelname']}: {data['message']}"
    if len(other_keys)>0:
        msg+= '\t\t' + other_keys
    return msg


class DebugLoggingWrap(LoggingWrap):
    def __init__(self, scope='tg', level=logging.DEBUG, add_keys = False):
        super(DebugLoggingWrap, self).__init__(
            scope,
            level,
            _debug_formatting if not add_keys else _debug_formatting_with_keys
        )
