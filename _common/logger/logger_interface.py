from typing import *
from logging import getLogger
from copy import deepcopy
from collections import Iterable

class LoggerInterface:
    def output(self, method, object):
        raise NotImplementedError()

    def set_extra_fields(self, fields: Dict):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class NullLoggerInterface(LoggerInterface):
    def output(self, method, object):
        pass

    def set_extra_fields(self, fields: Dict):
        pass

    def close(self):
        pass