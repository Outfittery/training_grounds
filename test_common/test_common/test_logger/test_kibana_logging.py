from unittest import TestCase
from tg.common._common.logger.kibana_logging_wrap import KibanaTypeProcessor
import datetime
import json

class Sc:
    def __init__(self, tc: TestCase):
        self.proc = KibanaTypeProcessor()
        self.history = []
        self.tc = tc

    def log(self, **kwargs) -> 'Sc':
        self.history.append(self.proc.process(kwargs))
        return self

    def check(self, **kwargs):
        self.tc.assertDictEqual(kwargs, self.history[-1])
        return self


class KibanaLoggingTestCase(TestCase):
    def test_primitives(self):
        (Sc(self)
         .log(i=5, b=True, f=0.1, s='ab')
         .check(i=5, b=True, f=0.1, s='ab')
         )

    def test_json(self):
        (Sc(self)
         .log(l=[3,4], d={'a':5})
         .check(l='[3, 4]', d='{"a": 5}')
         )

    def test_datetime(self):
        dt = datetime.datetime(2020,1,1,12,00)
        (Sc(self)
         .log(dt=dt)
         .check(dt = '2020-01-01 12:00:00')
        )

    def test_none(self):
        (Sc(self)
         .log(n=None)
         .check()
         )

    def test_none_replacement(self):
        dt = datetime.datetime(2020, 1, 1, 12, 00)
        (
            Sc(self)
            .log(i=None, b=None, f=None, s=None, l=None, d=None, dt=None)
            .check()
            .log(i=5, b=True, f=0.1, s='ab', l=[1], d={'a':5}, dt = dt)
            .log(i=None, b=None, f=None, s=None, l=None, d=None, dt=None)
            .check(i=-1, b=False, f=0, s='', l='null', d='null', dt='')
        )
