from unittest import TestCase
from tg.common.misc import AlgorithmStack

class Success1:
    def run(self, x):
        return x+1


class Success2:
    def run(self, x):
        return x-1

class Failure1:
    def run(self, x):
        raise ValueError('F1')


class Failure2:
    def run(self, x):
        raise ValueError('F2')

class AlgorithmBatteryTestCase(TestCase):
    def test_simple(self):
        result = AlgorithmStack().add(Success1().run).add(Success2().run).add(Failure1().run).run(1)
        self.assertEqual(1,len(result.history))
        self.assertEqual(2,result.result)



    def test_first_failure(self):
        result = AlgorithmStack().add(Failure1().run).add(Success2().run).run(2)
        self.assertEqual(2, len(result.history))
        for key, value in result.history[0].items():
            self.assertIsInstance(value,(str,bool))
        self.assertEqual(1, result.result)


    def test_two_failures(self):
        result =AlgorithmStack().add(Failure1().run).add(Failure2().run).add(Success2().run).run(2)
        self.assertEqual(3, len(result.history))
        self.assertEqual(1, result.result)
        print(result.history)


