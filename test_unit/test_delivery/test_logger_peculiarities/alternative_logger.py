from ...._common.logger import LoggerRoot as _LoggerRoot

class LoggingWrapInstance(_LoggerRoot):
    def __init__(self):
        super(LoggingWrapInstance, self).__init__()

Logger = LoggingWrapInstance()
