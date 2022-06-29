from unittest import TestCase
from tg.common.datasets.selectors import *
from pprint import pprint
import json

data = {
    'a': {
        'x':
            {
                'u': 1
            },
        'y': {
            'v': 2,
            'w': 3
        }
    },
    'b': 5,
    'c': 6,
    'd': None
}


class SelectorTestCase(TestCase):
    def assertSelect(self, value, selector, pr=False):
        if pr:
            print(json.dumps(selector.simple_repr(), indent=2))
        result = selector(data)
        if isinstance(value, dict):
            self.assertDictEqual(value, result)
        else:
            self.assertEqual(value, result)

    def test_simple(self):
        self.assertSelect(
            {'b': 5},
            Selector().select('b')
        )

    def test_several_addrs(self):
        self.assertSelect(
            {'b': 5, 'c': 6},
            Selector().select('b', 'c')
        )

    def test_long_addr(self):
        self.assertSelect(
            {'b': 5, 'u': 1, 'w': 3},
            Selector().select('a.x.u', 'a.y.w', 'b')
        )

    def test_prefixed_addr(self):
        self.assertSelect(
            {'v': 2, 'w': 3},
            Selector().with_prefix('a.y').select('v', 'w')
        )

    def test_two_part_selector_1(self):
        self.assertSelect(
            {'b': 5, 'v': 2, 'w': 3},
            Selector().with_prefix('a.y').select('v', 'w').select('b')
        )

    def test_two_part_selector_2(self):
        self.assertSelect(
            {'b': 5, 'v': 2, 'w': 3},
            Selector().select('b').with_prefix('a.y').select('v', 'w')
        )

    def test_two_part_selector_3(self):
        self.assertSelect(
            {'u': 1, 'v': 2, 'w': 3},
            Selector().with_prefix('a.x').select('u').with_prefix('a.y').select('v', 'w')
        )

    def test_selector_with_feed(self):
        self.assertSelect(
            {'b': '5'},
            Selector().select(['b', FunctionFeed(str)])
        )

    def test_none_propagation(self):
        self.assertSelect(
            {'xx': None},
            Selector().select('d.x.xx')
        )

    def test_empty_as_none(self):
        self.assertSelect(
            {'xx': None},
            Selector().select('xx.xx.xx')
        )

    def test_selector_with_function(self):
        def f(v):
            if isinstance(v, str):
                return v + '1'
            else:
                return v - 1
        self.assertSelect(
            {'b': 4, 'x': '51', 'u': 1},
            Selector().select(['b', f], x=['b', str, f]).with_prefix('a.x').select('u')
        )
