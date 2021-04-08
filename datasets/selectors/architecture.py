from typing import *



def _get_selector_name(selector) -> Optional[str]:
    if isinstance(selector, str):
        return selector
    if isinstance(selector, CombinedSelector):
        return selector.__repr__()
    else:
        try:
            return selector.__name__
        except:
            return selector.__repr__()


class SelectorCallStackItem:
    """
    Representation of one call in the selectors' callstack
    """
    def __init__(self, selector_type: str, selector_name: str, stage: Union[str,int]):
        """

        Args:
            selector_type: Python type of the calling selector
            selector_name: Name of the calling selector
            stage: The description of the moment in calling selector's algorithm when the call was made
        """
        self.selector_type = selector_type
        self.selector_name = selector_name
        self.stage = stage

    def __repr__(self):
        return self.__dict__.__repr__()


class SelectorCallStack:
    """
    Selectors' call stack.
    Contains the callstack of selectors (which are of :class:``CombinedSelector`` type), and the
    last call (which can be an arbitrary callable)
    """
    def __init__(self):
        """
        Creates an empty call stack. To create a meaningful call stack, :func:``append_call`` is used.
        """
        self.call_stack = ()  # type: Optional[Tuple]
        self.called_object_name = None



    def append_call(self, selector: Any, stage: Union[str,int], called_object: Any) -> 'SelectorCallStack':
        """
        Appends a call to the call stack
        Args:
            selector: the selector that makes a call
            stage: The description of the moment in calling selector's algorithm when the call was made
            called_object: the callable that was called (maybe another selector, or an arbitrary function)

        Returns:

        """
        ch = SelectorCallStack()
        ch.call_stack = self.call_stack + (SelectorCallStackItem(type(selector).__name__,
                                                                 _get_selector_name(selector),
                                                                 stage),)
        ch.called_object_name = _get_selector_name(called_object)
        return ch

    
class SelectorDataPathContext:
    """
    An internal class that is used to build the data path
    """
    def __init__(self, in_chain: Tuple):
        self.in_chain = in_chain
        self.out_chain = None


class SelectionRootInfo:
    def __init__(self, name: Optional[str], id_selector: Optional[Callable]):
        self.name = name
        self.id_selector = id_selector


class SelectionContext:
    """
    Represents the overall context of selection process.
    """
    @staticmethod
    def create(original_object: Any, root_info: SelectionRootInfo) -> 'SelectionContext':
        """
        Creates a selection context out of the object that is subjected to featurization
        Args:
            original_object:
        """
        self = SelectionContext()
        self.call_stack = SelectorCallStack()
        self.warnings = []
        self.chain_context = SelectorDataPathContext(tuple())
        self.original_object = original_object
        self.root_info = root_info
        if root_info.id_selector is not None:
            self.original_object_id = root_info.id_selector(original_object)
        else:
            self.original_object_id = None
        return self

    def __init__(self):
        """
        This constructor is not meant to be used publically
        """
        self.original_object = None
        self.call_stack = None # type:Optional[SelectorCallStack]
        self.warnings = None # type: Optional[List]
        self.chain_context = SelectorDataPathContext(tuple()) #type: SelectorDataPathContext


    def append_call(self, selector: Any, stage: Union[str,int], called_object: Any, data_path: Tuple = tuple()) -> 'SelectionContext':
        """
        Adds a call to the context

        Args:
            selector: the selector that makes a call
            stage: The description of the moment in calling selector's algorithm when the call was made
            called_object: the callable that was called (maybe another selector, or an arbitrary function)
            data_path: the current data path

        Returns:

        """
        ctx = SelectionContext()
        ctx.call_stack = self.call_stack.append_call(selector, stage, called_object)
        ctx.warnings = self.warnings
        ctx.chain_context.in_chain = self.chain_context.in_chain+data_path
        ctx.chain_context.out_chain = None
        ctx.original_object = self.original_object
        return ctx

    def get_data_path(self) -> str:
        """
        Generates string representation of the data path, that was processed on the moment of this context
        """
        return '.'.join(self.chain_context.in_chain)


    def get_code_path(self) -> str:
        """
        Gets the string representation of the call stack on the moment of this context

        """
        code_path = ''
        for cstack in self.call_stack.call_stack:
            code_path += '/' + str(cstack.stage)
        code_path += ':' + str(self.call_stack.called_object_name)
        return code_path



class SelectorException(Exception):
    """
    Exception describing the error occured somewhere in the selection.
    Contains the context that was actual at the time of the error, thus enabling
    error tracing by code path (the location of the erroneous selector in the graph)
    and data path (the location of data that triggered an error within the input data)
    """
    def __init__(self, context: SelectionContext):
        self.context = context
        super(SelectorException, self).__init__(self._generate_message())

    def _generate_message(self):
        try:
            return self.context.get_code_path()+"\n"+self.context.get_data_path()
        except:
            return 'FAILED TO PRODUCE MESSAGE FOR FEATURIZATION CALLSTACK'



class CombinedSelector:
    """
    Abstract class for the selector, that enables the compositions of other selectors and traces errors in their structure
    """
    def __init__(self):
        self._name = None  # type: Optional[str]
        self._id_selector = None #type: Optional[Callable]

    def assign_name(self, name: str) -> 'CombinedSelector':
        """
        The specific name can be assigned to the selector. This feature is mostly ignored right now

        Returns: self for FluentAPI

        """
        self._name = name
        return self

    def assign_id_selector(self, id_selector: Callable):
        self._id_selector = id_selector
        return self

    def __call__(self, obj: Any, context: Optional[SelectionContext] = None) -> Any:
        if context is None:
            context = SelectionContext.create(obj, SelectionRootInfo(self._name, self._id_selector))
        return self._internal_call(obj, context)

    def call_and_return_context(self, obj: Any) -> Tuple[Any, SelectionContext]:
        """
        Performs the call and returns not only the result, but also the context. The context will contain warnings.

        """
        context = SelectionContext.create(obj, SelectionRootInfo(self._name, self._id_selector))
        return self(obj,context), context

    def _internal_call(self, obj: Any, context: SelectionContext):
        """
        Must be implemented by descendants
        """
        raise NotImplementedError()

    def _further_call(self, obj: Any, context: SelectionContext, selector: Callable):
        """
        Must be called by descendants.
        This method will call the selector and arrange all the nesessary error handling

        """
        if not isinstance(selector, CombinedSelector):
            try:
                return selector(obj)
            except Exception as e:
                raise SelectorException(context)
        else:
            try:
                return selector(obj, context)
            except SelectorException as se:
                raise se
            except Exception as e:
                raise SelectorException(context) from e

    def __repr__(self):
        if self._name is not None:
            return self._name
        return type(self).__name__

    def get_structure(self):
        return None
