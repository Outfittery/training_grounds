from unittest import TestCase
from tg.common import Logger
from tg.common._common.logger import LogBuffer
from tg.common._common.logger.debug_logging_wrap import DebugLoggingWrap


class DebugDefaultLoggingTestCase(TestCase):
    def test_default(self):
        buffer = LogBuffer(custom_wrap_factory=DebugLoggingWrap)
        Logger.debug('test0')
        Logger.info('test1')
        Logger.warning('test2')
        Logger.error('test3')
        data = buffer.read().select(lambda z: z.split(' ')).to_list()
        expected = [('DEBUG', 'test0'), ('INFO', 'test1'), ('WARNING', 'test2'), ('ERROR', 'test3')]
        print(data)
        for d, e in zip(data, expected):
            self.assertEqual(e[0] + ":", d[2])
            self.assertEqual(e[1], d[3])
