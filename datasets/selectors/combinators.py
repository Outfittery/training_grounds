from typing import *
from .architecture import CombinedSelector, SelectionContext, SelectorException

from .architecture import _get_selector_name
from ..._common import OldTGWarning



# TODO: add short description of module
# TODO: make imports explicit
# TODO: move original object from history to context


class AbstractPipeline(CombinedSelector):
    """
    Abstract class for pipeline-like selectors
    """

    def __init__(self):
        super(AbstractPipeline, self).__init__()

    def _get_selectors(self, obj):
        """
        Descendants must return the list of selectors to be applied to the given object
        """
        raise NotImplementedError()

    def get_structure(self):
        return {key:value for key, value in self._get_selectors(None)}

    def _internal_call(self, obj, context: SelectionContext):
        result_chain = tuple()
        selectors = self._get_selectors(obj)
        for index, selector in selectors:
            further_context = context.append_call(self, index, selector, result_chain)
            obj = self._further_call(obj, further_context, selector)
            if further_context.chain_context.out_chain is None:
                result_chain += (_get_selector_name(selector),)
            else:
                result_chain += further_context.chain_context.out_chain
        context.chain_context.out_chain = result_chain
        return obj


class Pipeline(AbstractPipeline):
    """
    Normal pipeline.
    Callables are called in chain, the result of the last call is returned
    """

    def __init__(self, *selectors: Callable):
        super(Pipeline, self).__init__()
        self.selectors = [(index, selector) for index, selector in enumerate(selectors)]

    def _get_selectors(self, obj):
        return self.selectors

    def get_structure(self):
        return {key: value for key, value in self.selectors}




class MergeException(Exception):
    """
    Exception that is thrown when Ensemble-like selectors cannot merge the output
    """

    def __init__(self, key):
        self.key = key
        super(MergeException, self).__init__(f"Duplicating key `{self.key}` on merging results")


class AbstractEnsemble(CombinedSelector):
    """
    An abstract class for ensemble-like
    """

    def __init__(self):
        super(AbstractEnsemble, self).__init__()

    def _get_selectors(self, obj) -> Tuple[List, Dict]:
        """Descendants must return named and unnamed selectors for a given object"""
        raise NotImplementedError()

    def _assemble_results(self, unnamed: List, named: Dict, context: SelectionContext) -> Any:
        """Descendants must assemble the results of named and unnamed selectors"""
        raise NotImplementedError()

    def _internal_call(self, obj, context: SelectionContext):
        unnamed, named = self._get_selectors(obj)

        selectors = (
                [(False, index, selector) for index, selector in enumerate(unnamed)] +
                [(True, key, selector) for key, selector in named.items()]
        )
        unnamed_result = []
        named_result = {}
        for named, key, selector in selectors:
            further_context = context.append_call(self, key, selector)
            res = self._further_call(obj, further_context, selector)
            if named:
                named_result[key] = res
            else:
                unnamed_result.append(res)
        context.chain_context.out_chain = (context.get_code_path(),)
        return self._assemble_results(unnamed_result, named_result, context)


class Ensemble(AbstractEnsemble):
    """
    Ensemble selector
    """

    def __init__(self, *selectors, **named_selectors):
        super(Ensemble, self).__init__()
        self.selectors = selectors
        self.named_selectors = named_selectors

    def _get_selectors(self, obj):
        return self.selectors, self.named_selectors

    def _assemble_results(self, unnamed: List, named: Dict, context):
        result = {}
        for index, res in enumerate(unnamed):
            for key in res:
                if key in result:
                    raise SelectorException(context.append_call(self, index, 'internal')) from MergeException(key)
                result[key] = res[key]
        for key, res in named.items():
            if key in result:
                raise SelectorException(context.append_call(self, key, 'internal')) from MergeException(key)
            result[key] = res
        return result

    def get_structure(self):
        result = {}
        for index, selector in enumerate(self.selectors):
            result[index]=selector
        for key, selector in self.named_selectors.items():
            result[key]=selector
        return result


class FieldGetter(CombinedSelector):
    """
    This selector retrieves a field from a dict-like object. Supports none-propagation
    """

    def __init__(self, field: Any, none_propagation=True):
        super(FieldGetter, self).__init__()
        self.field = field
        self.none_propagation = none_propagation
        self.name = f'[{"?" if self.none_propagation else ""}{self.field}]'

    def _internal_call(self, obj, context: SelectionContext):
        if not self.none_propagation:
            return obj[self.field]
        else:
            if isinstance(obj, dict) and self.field in obj:
                return obj[self.field]
            if isinstance(obj, list) and isinstance(self.field, int) and 0 <= self.field < len(obj):
                return obj[self.field]
            context.warnings.append(OldTGWarning(
                "Missing field {field},  code_path {code_path}, data path {data_path}",
                field=self.field,
                code_path=context.get_code_path(),
                data_path=context.get_data_path()
            ))
            return None

    def __repr__(self):
        return self.name


class FunctionFeed(CombinedSelector):
    """
    This selector represents a step in the pipeline, when the currently retrieved object is fed to the function.
    Supports none-propagation
    """

    def __init__(self, callable: Callable, none_propagation=True):
        super(FunctionFeed, self).__init__()
        self.callable = callable
        self.none_propagation = none_propagation
        self._name = f'({"?" if self.none_propagation else ""}{self.callable})'

    def _internal_call(self, obj, context: SelectionContext):
        if self.none_propagation:
            if obj is None:
                context.warnings.append(OldTGWarning(
                    "None argument for Feed to {callable}, code path {code_path}, data path {data_path}",
                    callable=str(self.callable),
                    code_path=context.get_code_path(),
                    data_path=context.get_data_path()
                ))
                return None
        return self.callable(obj)

    def __repr__(self):
        return self._name


class Listwise(AbstractEnsemble):
    """
    Applies the same selector to all the elements of the list
    """

    def __init__(self, selector):
        super(Listwise, self).__init__()
        self.selector = selector

    def _build_pipeline(self, index):
        return Pipeline(FieldGetter(index), self.selector)

    def _get_selectors(self, obj):
        return [], {index: self._build_pipeline(index) for index in range(len(obj))}

    def _assemble_results(self, unnamed: List, named: Dict, context: SelectionContext):
        return [named[index] for index in range(len(named))]

    def get_structure(self):
        return {'*': self._build_pipeline('*')}



class Dictwise(AbstractEnsemble):
    """
    Applied the same selector to all the elements of the dictionary
    """

    def __init__(self, selector):
        super(Dictwise, self).__init__()
        self.selector = selector

    def _build_pipeline(self, key):
        return Pipeline(FieldGetter(key), self.selector)

    def _get_selectors(self, obj):
        return [], {key: self._build_pipeline(key) for key in obj}

    def _assemble_results(self, unnamed: List, named: Dict, context: SelectionContext):
        return named

    def get_structure(self):
        return {'*': self._build_pipeline('*')}




def transpose_list_of_dicts_to_dict_of_lists(obj):
    fields = dict()  # type: Dict[str,List]
    for index, dct in enumerate(obj):
        for key, value in dct.items():
            if key not in fields:
                fields[key] = [None for i in range(index)]
            fields[key].append(value)
        for key in fields.keys():
            if key not in dct:
                fields[key].append(None)
    return fields


class ListFeaturizer(AbstractPipeline):
    """
    Non-stable.
    """

    def __init__(self,
                 list_element_to_dict: Callable,
                 dict_fields_to_value: Callable
                 ):
        super(ListFeaturizer, self).__init__()
        self.list_element_to_dict = list_element_to_dict
        self.dict_fields_to_value = dict_fields_to_value

    def _get_selectors(self, obj):
        return [
            ('list_element_to_dict', Listwise(self.list_element_to_dict)),
            ('transpose', transpose_list_of_dicts_to_dict_of_lists),
            ('dict_fields_to_value', Dictwise(self.dict_fields_to_value))
        ]

    def get_structure(self):
        return self._get_selectors(None)


