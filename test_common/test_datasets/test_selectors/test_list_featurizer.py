from unittest import TestCase
from tg.common.datasets.selectors import ListFeaturizer, SelectorException

def element_to_dict(element):
    result = {d: int(element[0]) for d in element[1:]}
    return result

def _throwing(*args,**kwargs):
    raise ValueError()

class ListFeaturizerTestCase(TestCase):
    def assertFeatures(self, result, *elements):
        ft = ListFeaturizer(
            element_to_dict,
            lambda z: z
        )
        self.assertDictEqual(
            result,
            ft(list(elements))
        )

    def test_simple(self):
        self.assertFeatures(
            {'a': [1, 2], 'b': [1, 2]},
            '1ab',
            '2ab'
        )

    def test_with_skipped_values(self):
        self.assertFeatures(
            {'a': [1, 2, None, 4, None], 'b': [1, 2, 3, None, None], 'c': [None, None, 3, 4, None]},
            '1ab',
            '2ab',
            '3bc',
            '4ac',
            '5'
        )

    def assertHistory(self, exp: SelectorException, index, type, name, stage):
        item = exp.context.call_stack.call_stack[index]
        self.assertEqual(type, item.selector_type)
        self.assertEqual(name, item.selector_name)
        self.assertEqual(stage, item.stage)

    def test_failure_in_aggregator(self):
        try:
            ft = ListFeaturizer(
                element_to_dict,
                _throwing
            )
            ft(['1a','2b'])
            self.fail('Did not throw')
        except SelectorException as exp:
            self.assertHistory(exp,0,'ListFeaturizer','ListFeaturizer','dict_fields_to_value')
            self.assertEqual(exp.context.call_stack.called_object_name,'_throwing')


    def test_failure_in_selector(self):
        try:
            ft = ListFeaturizer(
                _throwing,
                lambda z: z
            )
            ft(['1a','2b'])
        except SelectorException as exp:
            self.assertHistory(exp, 0, 'ListFeaturizer', 'ListFeaturizer', 'list_element_to_dict')
            self.assertEqual(exp.context.call_stack.called_object_name,'_throwing')