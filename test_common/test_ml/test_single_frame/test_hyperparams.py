from tg.common.ml.training_core._hyperparams import _apply_hyperparams as apply_hyperparams
from unittest import TestCase


class C:
    def __init__(self):
        self.a = [1, 2]
        self.b = {
            'x': [10, 11],
            'y': '_'
        }


class ApplyHyperparameterTestCase(TestCase):
    def assertAssignment(self, expected_value, key, getter, value=None):
        obj = C()
        if value is None:
            value = str(expected_value)
        apply_hyperparams({key: value}, obj)
        self.assertEqual(expected_value, getter(obj))

    def test_field_no_type(self):
        self.assertAssignment('abc', 'a', lambda z: z.a)

    def test_field_int_type(self):
        self.assertAssignment(15, 'a:int', lambda z: z.a)

    def test_field_float_type(self):
        self.assertAssignment(0.25, 'a:float', lambda z: z.a)

    def test_array(self):
        self.assertAssignment('xx', 'a.0', lambda z: z.a[0])

    def test_dict(self):
        self.assertAssignment('xx', 'b.x', lambda z: z.b['x'])

    def test_array_inside_dict(self):
        self.assertAssignment('xx', 'b.x.0', lambda z: z.b['x'][0])

    def test_wrong_array_index_fails(self):
        self.assertRaises(ValueError, lambda: self.assertAssignment('xx', 'a.2', lambda z: z.a[2]))

    def test_wrong_dict_field_doesnt_fail(self):
        self.assertAssignment('xx', 'b.z', lambda z: z.b['z'])

    def test_wrong_obj_field_fails(self):
        self.assertRaises(ValueError, lambda: self.assertAssignment('xx', 'c', lambda z: z.c))
