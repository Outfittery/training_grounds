from unittest import TestCase
from tg.common.datasets.selectors import *


class IdSelectorTestCase(TestCase):
    def test_id_selector(self):
        id_selector = lambda obj: obj['main']['id']
        main_selector = Selector().select('a.a', 'b.b').assign_id_selector(id_selector)
        obj = dict(a=dict(a=1), b=dict(b=2), main=dict(id='xx'))
        result, context = main_selector.call_and_return_context(obj)
        self.assertDictEqual(dict(a=1, b=2), result)
        self.assertEqual('xx', context.original_object_id)
