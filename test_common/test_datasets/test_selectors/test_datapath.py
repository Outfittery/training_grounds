from unittest import TestCase
from tg.common.datasets.selectors import *


def _throwing(obj):
    raise ValueError()


class proc:
    def __init__(self, name):
        self.name = name

    def __call__(self, x):
        return x

    def __repr__(self):
        return self.name


def _init(_):
    return dict(a=1, b=2)


class DataPathTestCase(TestCase):
    def assertDataPath(self, pipeline, *address):
        try:
            pipeline(0)
            self.fail('Pipeline did not throw')
        except SelectorException as ex:
            if len(address) == 0:
                print(ex.context.chain_context.in_chain)
            self.assertListEqual(list(address), list(ex.context.chain_context.in_chain))
        except:
            self.fail('Raised wrong exception')

    def test_pipeline(self):
        pipe = Pipeline(proc('a'), proc('b'), _throwing)
        self.assertDataPath(pipe, 'a', 'b')

    def test_ensemble(self):
        self.assertDataPath(
            Pipeline(
                proc('a'),
                Ensemble(
                    x1=Pipeline(proc('b'), proc('c')),
                    x2=Pipeline(proc('d'), _throwing))
            ),
            'a', 'd'
        )

    def test_ensemble_in_the_middle(self):
        pipe = Pipeline(
            Ensemble(x1=proc('a'), x2=proc('b')),
            Ensemble(y1=proc('c'), y2=proc('d')),
            proc('e'),
            Pipeline(proc('f'), _throwing)
        )
        self.assertDataPath(
            pipe,
            '/0:Ensemble',
            '/1:Ensemble',
            'e',
            'f'
        )

    def test_selector(self):
        self.assertDataPath(
            Pipeline(
                _init,
                Selector().select(
                    a='a',
                    b=['b', float, _throwing]
                )
            ),
            '_init',
            '[?b]',
            "(?<class 'float'>)"
        )
