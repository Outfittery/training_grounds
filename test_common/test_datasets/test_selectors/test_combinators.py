from unittest import TestCase
from tg.common.datasets.selectors import *


def _throwing(obj):
    raise ValueError()


class CombinatorsTestCase(TestCase):
    def test_pipeline(self):
        self.assertEqual(
            6,
            Pipeline(lambda z: z * 2, lambda z: z + 2)(2)
        )

    def test_ensemble(self):
        self.assertDictEqual(
            {'a': 1, 'b': 2, 'c': 3},
            Ensemble(lambda z: {'a': z}, lambda z: {'b': z + 1}, c=lambda z: z + 2)(1)
        )

    def test_combination(self):
        self.assertDictEqual(
            {'a': 1},
            Ensemble(Pipeline(Ensemble(lambda z: {'a': z})))(1)
        )

    def assertHistory(self, exp: SelectorException, index, type, name, stage):
        item = exp.context.call_stack.call_stack[index]
        self.assertEqual(type, item.selector_type)
        self.assertEqual(name, item.selector_name)
        self.assertEqual(stage, item.stage)

    def getExp(self, featurizer, inner_type=None):
        try:
            featurizer(1)
        except Exception as e:
            self.assertIsInstance(e, SelectorException)
            self.assertEqual(1, e.context.original_object)
            if inner_type is not None:
                self.assertIsInstance(e.__cause__, inner_type)
            return e
        self.fail('Did not throw')

    def test_error_in_pipeline(self):
        exp = self.getExp(Pipeline(lambda z: z + 1, lambda z: z + 2, _throwing).assign_name('x'))
        self.assertHistory(exp, 0, 'Pipeline', 'x', 2)
        self.assertEqual(exp.context.call_stack.called_object_name, '_throwing')

    def test_error_in_ensemble_index(self):
        exp = self.getExp(Ensemble(lambda z: {}, _throwing).assign_name('y'))
        self.assertHistory(exp, 0, 'Ensemble', 'y', 1)
        self.assertEqual(exp.context.call_stack.called_object_name, '_throwing')

    def test_error_in_ensemble_key(self):
        exp = self.getExp(Ensemble(a=lambda z: {}, b=_throwing))
        self.assertHistory(exp, 0, 'Ensemble', 'Ensemble', 'b')
        self.assertEqual(exp.context.call_stack.called_object_name, '_throwing')

    def test_error_in_pipeline_internal(self):
        pip = Ensemble(b=Pipeline(lambda z: z + 1, lambda z: z + 2))
        pip.named_selectors['b'].selectors = None
        exp = self.getExp(pip)
        self.assertHistory(exp, 0, 'Ensemble', 'Ensemble', 'b')
        self.assertEqual(exp.context.call_stack.called_object_name, 'Pipeline')

    def test_merge_index(self):
        exp = self.getExp(Ensemble(lambda z: {'a': 1}, lambda z: {'a': 2}), MergeException)
        self.assertHistory(exp, 0, 'Ensemble', 'Ensemble', 1)
        self.assertEqual('a', exp.__cause__.key)

    def test_merge_key(self):
        exp = self.getExp(Ensemble(lambda z: {'a': 1}, a=lambda z: z), MergeException)
        self.assertHistory(exp, 0, 'Ensemble', 'Ensemble', 'a')
        self.assertEqual('a', exp.__cause__.key)

    def test_exception_root(self):
        pip = Pipeline(lambda z: z + 1)
        pip.selectors = None
        self.assertRaises(TypeError, lambda: pip(0))

    def test_listwise(self):
        ppl = Listwise(str)
        self.assertListEqual(
            ['0', '1'],
            ppl([0, 1])
        )

    def test_dictwise(self):
        ppl = Dictwise(str)
        self.assertDictEqual(
            dict(a='4', b='5'),
            ppl(dict(a=4, b=5))
        )
