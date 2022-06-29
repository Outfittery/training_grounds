from unittest import TestCase
from threading import Thread
import time
from tg.common._common.logger.logger_root import Logger
from tg.common._common.logger.logging_wrap import LoggingWrap
from tg.common.test_common.test_common.test_logger.test_fields_gathering import LogBuffer as _LogBuffer
from copy import deepcopy
import pandas as pd


class SlowLogBuffer(_LogBuffer):
    def _serialize(self, data):
        time.sleep(0.001)
        result = super(SlowLogBuffer, self)._serialize(data)
        time.sleep(0.001)
        return result


class Worker:
    def __init__(self, prefix, n=100):
        self.prefix = prefix
        self.n = n

    def run(self):
        for i in range(self.n):
            Logger.push_keys(**{self.prefix + '_' + str(i): str(i)})
            time.sleep(0.001)
            Logger.info(self.prefix + '_' + str(i), step=i, thread=self.prefix)
            time.sleep(0.001)


class MultithreadingTestCase(TestCase):
    def _check_df(self, df, thread, step, value=None):
        column = thread + '_' + str(step)
        cl = df[column]
        if value is None:
            error = df.loc[~cl.isnull()]
        else:
            error = df.loc[cl != value]
        if error.shape[0] > 0:
            print(df[['message', 'thread', 'step', column]].assign(expexted_value=value))
        self.assertEqual(0, error.shape[0])

    def test_multithreading(self):
        M = 5
        N = 100
        buffer = SlowLogBuffer()
        threads = []
        for i in range(M):
            worker = Worker('Thread' + str(i), N)
            thread = Thread(target=worker.run)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        df = buffer.parse().to_dataframe()
        a_columns = [c for c in df if c.startswith('ThreadA')]
        b_columns = [c for c in df if c.startswith('ThreadB')]

        pd.options.display.width = None

        for j in range(M):
            thread = 'Thread' + str(j)
            for i in range(N):
                self._check_df(df.loc[df.thread != thread], thread, i)
                self._check_df(df.loc[(df.thread == thread) & (df.step < i)], thread, i)
                self._check_df(df.loc[(df.thread == thread) & (df.step >= i)], thread, i, str(i))
