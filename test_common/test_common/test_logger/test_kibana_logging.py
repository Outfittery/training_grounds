from unittest import TestCase
from tg.common._common.logger.kibana_logging_wrap import KibanaTypeProcessor
import datetime
import json
import re

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

    def test_lists(self):
        (Sc(self)
         .log(l1=[3, 4], l2=[])
         .check(l1=['3', '4'], l2='')
         )
        
    def test_dicts(self):
        (Sc(self)
         .log(d1={'a': 5}, d2={}, d3={'a': [1, 2]})
         .check(d1={'a': 5}, d2={}, d3={'a': ['1', '2']})
         )
        
    def test_nested_dicts(self):
        (Sc(self)
         .log(d1={'a': {}}, d2={'a': {'b': 1}}, d3={'a': {'b': [1, 2]}}, d4={'a': {'b': {'c': [1]}}}, d5={'a': {1: 2}})
         .check(d1={'a': {}}, d2={'a': {'b': 1}}, d3={'a': {'b': ['1', '2']}}, d4={'a': {'b': {'c': ['1']}}}, d5={'a': {1: 2}})
         )

    def test_datetime(self):
        dt = datetime.datetime(2020, 1, 1, 12, 0, 1, 123456)
        sc = Sc(self)
        sc.log(dt=dt)
        val = sc.history[-1]['dt']
        print(val)
        self.assertIsNotNone(re.match('^2020-01-01T\d\d:00:01.123456\+00:00$', val))



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
            .log(i=5, b=True, f=0.1, s='ab', l=[1], d={'a': 5}, dt=dt)
            .log(i=None, b=None, f=None, s=None, l=None, d=None, dt=None)
            .check(i=-1, b=False, f=0.0, s='', l='', d={}, dt='')
        )

    def test_nested_none_replacement(self):
        (
            Sc(self)
            .log(d1=None, d2=None)
            .check()
            .log(d1={'a': {'b': 0.1}}, d2={'a': {'b': 1}}, d3={'a': {'b': {'c': [1]}}}, d4={'a': {'b': {'c': True}}})
            .log(d1={'a': None}, d2={'a': {'b': None}}, d3={'a': {'b': {'c': None}}}, d4={'a': {'b': {'c': None}}})
            .check(d1={'a': {}}, d2={'a': {'b': -1}}, d3={'a': {'b': {'c': ''}}}, d4={'a': {'b': {'c': False}}})
        )
