from typing import *
from .combinators import CombinedSelector, FieldGetter, Ensemble, FunctionFeed, Pipeline, SelectionContext



# TODO: add short description of module
# TODO: make imports explicit
# TODO: Selector must be an AbstractEnsemble and not CombinedSelector


class Selector(CombinedSelector):
    """
    Builder for featurization pipeline in the special case when the featurization is reduced to field extraction.
    Follows FluentAPI design.
    """
    def __init__(self, none_propagation = True):
        """

        Args:
            none_propagation: if True, None-propagation will be enabled by default in the constructed pipeline
        """
        super(Selector, self).__init__()
        self._current_prefix = None
        self._ensemble = Ensemble()
        self._none_propagation = none_propagation

    def with_prefix(self, *prefix: Union[str, FieldGetter]) -> 'Selector':
        """
        Sets the prefix for the next :meth:``Selector.select`` operation.

        Args:
            *prefix: string that contains one or many field names separated by dot
            OR :class:``FieldGetter`` object that represents an access to the field (in case if field name contains a dot)

        Returns: self

        """
        self._current_prefix = prefix
        return self

    def select(self, *args, **kwargs) -> 'Selector':
        """
        Adds selection in the pipeline
        Args:
            *args: string that contains one or many field names separated by dot;
            or array, containing either such strings or callables, in this case, the pipeline is denoted.
            the name of the field in the resulting dictionary is determined automatically.

            **kwargs:
            keys are names of the field in the resulting dictionary. Values are the same as *args


        Returns:

        """
        selector = self._select_to_ensemble(self._current_prefix, args, kwargs)
        self._ensemble.selectors += selector,
        self._current_prefix = None
        return self

    def _addr_to_transformer_list(self, addr):
        if isinstance(addr, tuple):
            addr = list(addr)
        elif not isinstance(addr, list):
            addr = [addr]
        addr_calls = []
        for index, item in enumerate(addr):
            if isinstance(item, str):
                for part in item.split('.'):
                    addr_calls.append(FieldGetter(part, self._none_propagation))
            elif isinstance(item, CombinedSelector):
                addr_calls.append(item)
            elif callable(item):
                addr_calls.append(FunctionFeed(item, self._none_propagation))
            else:
                raise ValueError(f'Address elements must be string instance, Selector or callable, but was {item}')
        return addr_calls

    def _get_last_field(self, pipe: List):
        for item in reversed(pipe):
            if isinstance(item, FieldGetter):
                return item.field
        raise ValueError(f'Cannot automatically determine the name of the field for address {pipe}')

    def _select_to_ensemble_without_prefix(self, args, kwargs):
        selectors = {}
        for key, addr in kwargs.items():
            selectors[key] = Pipeline(*self._addr_to_transformer_list(addr))
        for addr in args:
            pipeline = self._addr_to_transformer_list(addr)
            key = self._get_last_field(pipeline)
            selectors[key] = Pipeline(*pipeline)
        return Ensemble(**selectors)


    def _select_to_ensemble(self, prefix, args, kwargs):
        downstream = self._select_to_ensemble_without_prefix(args, kwargs)
        if prefix is None:
            return downstream
        return Pipeline(
            Pipeline(*self._addr_to_transformer_list(prefix)),
            downstream
        )

    def _internal_call(self, obj, context: SelectionContext):
        return self._ensemble(obj, context)


    @staticmethod
    def identity(x):
        return x


    def get_structure(self):
        return self._ensemble.get_structure()
