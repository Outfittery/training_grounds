from typing import *

import threading

from copy import deepcopy

from .logger_interface import LoggerInterface, NullLoggerInterface
from .debug_logging_wrap import DebugLoggingWrap
from .kibana_logging_wrap import KibanaLoggingWrap


_logger_lock = threading.Lock()


class LoggerRootBase:

    _instances = []

    def __init__(self):
        self._wrap = None  # type: Optional[LoggerInterface]
        self.base_keys = {}
        self.session_keys_store = {}
        self.initialize_default()

    def get_session_keys(self):
        thread_name = threading.current_thread().name
        if thread_name not in self.session_keys_store:
            self.session_keys_store[thread_name] = {}
        return self.session_keys_store[thread_name]

    def set_session_keys(self, value):
        thread_name = threading.current_thread().name
        self.session_keys_store[thread_name] = value

    def _internal_reset(self, wrap: LoggerInterface, keys: Dict = None):
        if self._wrap is not None:
            self._wrap.close()
        self._wrap = wrap
        self.base_keys = deepcopy(keys) if keys is not None else {}
        self.session_keys_store = {}
        self._update_keys(remove=True)

    def reset(self, wrap: LoggerInterface, keys: Dict = None):
        if self not in LoggerRootBase._instances:
            LoggerRootBase._instances.append(self)
        for ins in LoggerRootBase._instances:
            if ins != self:
                ins._internal_reset(None)
        self._internal_reset(wrap, keys)

    def initialize_default(self):
        raise NotImplementedError()

    def _merge_keys(self, *key_arrays):
        result = {}
        for key_array in key_arrays:
            for key, value in key_array.items():
                result[key] = value
        return result

    def _update_keys(self,
                     remove: Union[bool, List[str], str] = False,
                     add: Optional[Dict] = None):
        if isinstance(remove, bool):
            if remove:
                self.set_session_keys({})
        elif isinstance(remove, str):
            del self.get_session_keys()[remove]
        elif isinstance(remove, Iterable):
            for key in remove:
                del self.get_session_keys()[key]
        else:
            raise ValueError(f"`remove` is expected to be True, False, str or List[str], but was {type(remove)}")
        if add is not None:
            for key, value in add.items():
                self.get_session_keys()[key] = value
        if self._wrap is not None:
            self._wrap.set_extra_fields(self._merge_keys(self.base_keys, self.get_session_keys()))

    def push_keys(self, **kwargs):
        self._update_keys(False, kwargs)

    def clear_keys(self):
        self._update_keys(True)

    def _output(self, method, object, keys):
        if self._wrap is None:
            return
        with _logger_lock:
            if len(keys) != 0:
                self._wrap.set_extra_fields(self._merge_keys(self.base_keys, self.get_session_keys(), keys))
            self._wrap.output(method, object)
            if len(keys) != 0:
                self._wrap.set_extra_fields(self._merge_keys(self.base_keys, self.get_session_keys()))

    def info(self, object, **keys):
        self._output('info', object, keys)

    def warning(self, object, **keys):
        self._output('warning', object, keys)

    def error(self, object, **keys):
        self._output('error', object, keys)

    def debug(self, object, **keys):
        self._output('debug', object, keys)


class LoggerRoot(LoggerRootBase):
    def initialize_default(self, add_keys = False):
        self.reset(DebugLoggingWrap(add_keys=add_keys))

    def initialize_kibana(self):
        self.reset(KibanaLoggingWrap())

    def disable(self):
        self.reset(NullLoggerInterface())


Logger = LoggerRoot()
