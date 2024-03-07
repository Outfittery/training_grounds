from typing import TypeVar, Generic
from abc import ABC, abstractmethod

TData = TypeVar('TData')

class IAccessor(Generic[TData], ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_data(self, **kwargs) -> TData:
        pass

    def cache(self):
        from .cache import Cache
        return Cache.source(self)