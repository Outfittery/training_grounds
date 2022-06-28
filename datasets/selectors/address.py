from .combinators import Pipeline, CombinedSelector, FieldGetter, FunctionFeed
from .architecture import SelectionContext


class Address(CombinedSelector):
    def __init__(self, *addr, none_propagation=False):
        super(Address, self).__init__()
        calls = Address._addr_to_transformer_list(addr, none_propagation)
        self.pipeline = Pipeline(*calls)

    def _internal_call(self, obj, context: SelectionContext):
        return self.pipeline(obj, context)

    def get_structure(self):
        return self.pipeline.get_structure()

    @staticmethod
    def _addr_to_transformer_list(addr, none_propagation):
        if isinstance(addr, tuple):
            addr = list(addr)
        elif not isinstance(addr, list):
            addr = [addr]
        addr_calls = []
        for index, item in enumerate(addr):
            if isinstance(item, str):
                for part in item.split('.'):
                    addr_calls.append(FieldGetter(part, none_propagation))
            elif isinstance(item, int):
                addr_calls.append(FieldGetter(item,none_propagation))
            elif isinstance(item, CombinedSelector):
                addr_calls.append(item)
            elif callable(item):
                addr_calls.append(FunctionFeed(item, none_propagation))
            else:
                raise ValueError(f'Address elements must be string instance, selector or callable, but was {item}')
        return addr_calls

    @staticmethod
    def on(obj, none_propagation = False):
        return _AddressOn(obj, none_propagation)

    @staticmethod
    def elvis(*args):
        return Address(*args, none_propagation=True)


class _AddressOn:
    def __init__(self, obj, none_propagation):
        self.obj = obj
        self.none_propagation = none_propagation

    def __call__(self, *args):
        addr = Address(*args, none_propagation=self.none_propagation)
        return addr(self.obj)
