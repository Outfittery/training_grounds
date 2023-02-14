from unittest import TestCase
from tg.common._common.logger.logger_root import Logger


class LoggerTestCase(TestCase):
    def test_no_init(self):
        Logger.info('Test')
