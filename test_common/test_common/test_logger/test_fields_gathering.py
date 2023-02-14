from unittest import TestCase
import sys
from io import StringIO
from tg.common._common.logger.log_buffer import LogBuffer
from tg.common._common.logger.logger_root import Logger
import json
import datetime
from yo_fluq_ds import Query
import logging


class LoggerTestCase(TestCase):
    def test_simple(self):
        buffer = LogBuffer()
        Logger.info('t1')
        p1 = buffer.parse().to_list()
        Logger.info('t2')
        p2 = buffer.parse().to_list()
        self.assertEqual(1, len(p1))
        self.assertEqual(2, len(p2))
        self.assertEqual('t1', p1[0]['message'])
        self.assertEqual('t1', p2[0]['message'])
        self.assertEqual('t2', p2[1]['message'])

    def test_reinit(self):
        buffer1 = LogBuffer()
        Logger.info('t1')
        buffer2 = LogBuffer()
        Logger.info('t2')
        self.assertEqual(1, buffer1.parse().count())
        self.assertEqual(1, buffer2.parse().count())

    def test_fields(self):
        buffer = LogBuffer(base_key='x')
        Logger.info('t1')
        Logger.push_keys(session_key='y')
        Logger.info('t2')
        Logger.push_keys(session_key_1='z')
        Logger.info('t3')
        Logger.clear_keys()
        Logger.info('t4')
        rs = buffer.transpose('base_key', 'session_key', 'session_key_1', 'message')
        self.assertDictEqual(
            {'base_key': ['x', 'x', 'x', 'x'], 'session_key': ['#', 'y', 'y', '#'], 'session_key_1': ['#', '#', 'z', '#'], 'message': ['t1', 't2', 't3', 't4']},
            rs
        )

    def test_compatibilty_with_python_logging(self):
        buffer = LogBuffer(base_key='x')
        Logger.info('t1')
        logger = logging.getLogger('tg')
        logger.info('t2')
        rs = buffer.transpose('base_key', 'message')
        self.assertDictEqual({'base_key': ['x', 'x'], 'message': ['t1', 't2']}, rs)

    def test_levels(self):
        buffer = LogBuffer()
        Logger.info('t1')
        Logger.warning('t2')
        Logger.error('t3')
        rs = buffer.transpose('levelname', 'message')
        self.assertDictEqual(
            {'levelname': ['INFO', 'WARNING', 'ERROR'], 'message': ['t1', 't2', 't3']},
            rs
        )

    def test_call_stack(self):
        buffer = LogBuffer()
        Logger.info('t1')
        val = buffer.parse().single()
        fname = val['path'].split('/')[-1]
        this_fname = __file__.split('/')[-1]
        self.assertIn('path_line', val)
        self.assertEquals(this_fname, fname)

    def test_call_stack_for_logger(self):
        buffer = LogBuffer()
        logger = logging.getLogger('tg')
        logger.info('t1')
        val = buffer.parse().single()
        fname = val['path'].split('/')[-1]
        this_fname = __file__.split('/')[-1]
        self.assertIn('path_line', val)
        self.assertEquals(this_fname, fname)

    def test_exception(self):
        buffer = LogBuffer()
        Logger.info('Before try-except')
        try:
            Logger.info('Before exception')
            raise ValueError('ERROR')
            Logger.info('Must not see it')
        except:
            Logger.info('Info exception')
            Logger.error('Error exception')
        Logger.info('After try-except')
        rs = buffer.transpose('message', 'exception_value', 'exception_type')
        self.assertDictEqual(
            {'message': ['Before try-except', 'Before exception', 'Info exception', 'Error exception', 'After try-except'],
             'exception_value': ['#', '#', 'ERROR', 'ERROR', '#'],
             'exception_type': ['#', '#', "<class 'ValueError'>", "<class 'ValueError'>", '#']
             },
            rs)
        self.assertIn('exception_details', buffer.parse().skip(2).first())

    def test_object_as_argument(self):
        buffer = LogBuffer()
        Logger.info(dict(a=1, b=2))
        self.assertEqual("{'a': 1, 'b': 2}", buffer.parse().single()['message'])
