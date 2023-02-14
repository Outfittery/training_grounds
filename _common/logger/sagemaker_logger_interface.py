from .logger_interface import LoggerInterface


class SagemakerLoggerInterface(LoggerInterface):
    def output(self, method, object):
        print(f'{method.upper()}: {object}', flush=True)

    def set_extra_fields(self, fields):
        pass

    def close(self):
        pass
