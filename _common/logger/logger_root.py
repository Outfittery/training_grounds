from typing import *

import threading

from copy import deepcopy

from .logger_interface import LoggerInterface, NullLoggerInterface
from .debug_logging_wrap import DebugLoggingWrap
from .kibana_logging_wrap import KibanaLoggingWrap
from .timer import Timer
import os
import contextvars

_logger_lock = threading.Lock()
session_keys_store: contextvars.ContextVar[Optional[Dict]] = contextvars.ContextVar('session_keys_store', default=None)
session_timer: contextvars.ContextVar[Optional[Timer]] = contextvars.ContextVar('timer', default=None)


class LoggerState:
    def __init__(self):
        self._wrap = None  # type: Optional[LoggerInterface]
        self.base_keys = {}
        self.reset(DebugLoggingWrap(add_keys=False))

    def get_session_timer(self):
        timer = session_timer.get()
        if not timer:
            timer = Timer()
            session_timer.set(timer)
        return timer

    def set_session_timer(self, value):
        session_timer.set(value)

    def get_session_keys(self):
        keys = session_keys_store.get()
        if not keys:
            keys = {}
            session_keys_store.set(keys)
        return keys

    def set_session_keys(self, value):
        session_keys_store.set(value)

    def delete_keys(self, keys: Iterable[str]):
        session_keys = self.get_session_keys()
        for key in keys:
            del session_keys[key]

    def set_keys(self, keys: Dict):
        session_keys = self.get_session_keys()
        for key, value in keys.items():
            session_keys[key] = value

    def _merge_keys(self, *key_arrays):
        result = {}
        for key_array in key_arrays:
            for key, value in key_array.items():
                result[key] = value
        return result

    def update_keys(self,
                     remove: Union[bool, List[str], str] = False,
                     add: Optional[Dict] = None):
        if isinstance(remove, bool):
            if remove:
                self.set_session_keys({})
        elif isinstance(remove, str):
            self.delete_keys([remove])
        elif isinstance(remove, Iterable):
            self.delete_keys(remove)
        else:
            raise ValueError(f"`remove` is expected to be True, False, str or List[str], but was {type(remove)}")
        if add is not None:
            self.set_keys(add)
        if self._wrap is not None:
            self._wrap.set_extra_fields(self._merge_keys(self.base_keys, self.get_session_keys()))

    def reset(self, wrap: LoggerInterface, keys: Dict = None):
        if self._wrap is not None:
            self._wrap.close()
        self._wrap = wrap
        self.base_keys = deepcopy(keys) if keys is not None else {}
        self.set_session_keys(None)
        self.set_session_timer(None)
        self.update_keys(remove=True)

    def output(self, method, object, keys):
        if self._wrap is None:
            return
        with _logger_lock:
            self._wrap.set_extra_fields(self._merge_keys(self.base_keys, self.get_session_keys(), keys))
            self._wrap.output(method, object)
            self._wrap.set_extra_fields(self._merge_keys(self.base_keys, self.get_session_keys()))


def _get_unique_logger_state() -> LoggerState:
    ENV_KEY = 'tg_logger_unique_state_location'
    if ENV_KEY not in os.environ:
        state = LoggerState()
        os.environ[ENV_KEY] = LoggerState.__module__
        return state
    else:
        import importlib
        module_name = os.environ[ENV_KEY]
        m = importlib.import_module(module_name)
        return getattr(m, 'TGLoggerState')


TGLoggerState = _get_unique_logger_state()


class LoggerRootBase:
    def __init__(self):
        self.state = TGLoggerState

    def get_session_keys(self):
        return self.state.get_session_keys()

    def set_session_keys(self, value):
        self.state.set_session_keys(value)

    def push_keys(self, **kwargs):
        self.state.update_keys(False, kwargs)

    def clear_keys(self):
        self.state.update_keys(True)

    def info(self, object, **keys):
        self.state.output('info', object, keys)

    def warning(self, object, **keys):
        self.state.output('warning', object, keys)

    def error(self, object, **keys):
        self.state.output('error', object, keys)

    def debug(self, object, **keys):
        self.state.output('debug', object, keys)

    def initialize_default(self, add_keys = False):
        self.state.reset(DebugLoggingWrap(add_keys=add_keys))

    def reset(self, logging_wrap, keys: Optional[Dict] = None):
        self.state.reset(logging_wrap, keys)

    @property
    def timer(self):
        return self.state.get_session_timer()


class LoggerRoot(LoggerRootBase):
    def initialize_kibana(self):
        self.state.reset(KibanaLoggingWrap())

    def disable(self):
        self.state.reset(NullLoggerInterface())


Logger = LoggerRoot()
