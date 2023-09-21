from typing import *
import os
from hashlib import md5
from pathlib import Path
import pandas as pd
import pickle
from .._common import Loc, DataBundle
from .accessor import IAccessor, TData
from ..datasets.access import CacheMode
from enum import Enum
from abc import ABC, abstractmethod



class FunctionalDataAccessor(IAccessor):
    def __init__(self, method: Callable, *args, **kwargs):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def get_name(self):
        return Cache.default_name(self.method.__name__, *self.args, **self.kwargs)

    def get_data(self):
        return self.method(*self.args, **self.kwargs)


class ICacheHandler(ABC):
    @abstractmethod
    def accepts_object(self, obj: Any) -> bool:
        pass

    @abstractmethod
    def get_extension(self) -> str:
        pass

    @abstractmethod
    def write(self, obj: Any, path: Path):
        pass

    @abstractmethod
    def read(self, path: Path):
        pass


class PandasParquetHandler(ICacheHandler):
    def accepts_object(self, obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame)

    def get_extension(self) -> str:
        return 'parquet'

    def write(self, obj: Any, path: Path):
        obj.to_parquet(path)

    def read(self, path: Path):
        return pd.read_parquet(path)


class DefaultPickleHandler(ICacheHandler):
    def accepts_object(self, obj: Any) -> bool:
        return True

    def get_extension(self) -> str:
        return 'pkl'

    def write(self, obj: Any, path: Path):
        with open(path,'wb') as file:
            pickle.dump(obj, file)

    def read(self, path):
        with open(path,'rb') as file:
            return pickle.load(file)


class DataBundleHandler(ICacheHandler):
    def accepts_object(self, obj: Any) -> bool:
        return isinstance(obj, DataBundle)

    def get_extension(self) -> str:
        return 'bundle-zip'

    def write(self, obj: Any, path: Path):
        obj.save_as_zip(path)

    def read(self, path: Path):
        return DataBundle.load(path)






class Cache(Generic[TData]):
    class Mode(Enum):
        """
        Contains the modes for the cacheable data source.
        `No` means the cache will not be used at all.
        `Default` means the cache will be read if available, and created otherwise
        `Use` means the cache will be read if available, but if not, the exception is going to be raised
        `Remake` means the cache will be recreated regardless of its existence
        """

        No = "no"
        Default = "default"
        Use = "use"
        Remake = "remake"

    def __init__(self, source: IAccessor[TData]):
        self._source = source #type: IAccessor
        self._custom_filename = None #type: Optional[Path]
        self._custom_handler = None #type: Optional[ICacheHandler]
        self._cache_mode = Cache.DEFAULT_CACHE_MODE

    def to(self, file_name: Union[str,Path]) -> 'Cache[TData]':
        if isinstance(file_name, Path):
            self._custom_filename = file_name

        elif isinstance(file_name, str):
            if '/' in file_name:
                self._custom_filename = Path(file_name)
            else:
                self._custom_filename = Cache.DEFAULT_FOLDER/file_name

        return self

    def via(self, cache_handler: ICacheHandler) -> 'Cache[TData]':
        self._custom_handler = cache_handler
        return self

    def mode(self, mode: Union['Cache.Mode', str, CacheMode] = 'default') -> 'Cache[TData]':
        if isinstance(mode, CacheMode):
            mode = mode.name.lower()
        self._cache_mode = Cache.Mode(mode)
        return self

    def _get_path_prefix(self) -> Path:
        prefix = self._custom_filename if self._custom_filename is not None else Cache.DEFAULT_FOLDER/self._source.get_name()
        os.makedirs(prefix.parent, exist_ok=True)
        return prefix

    def _get_path(self) -> Optional[Path]:
        prefix = self._get_path_prefix()
        files = [f for f in os.listdir(prefix.parent) if f.startswith(prefix.name+'.')]
        if len(files)==0:
            return None
        if len(files)>1:
            raise ValueError(f'For the cache {prefix} two versions of cache exist. Clean manually.')
        return prefix.parent/files[0]

    def _from_cache(self) -> TData:
        path = self._get_path()
        if path is None:
            raise ValueError(f"Path was none when reading from cache: it's implementation problem")
        extension = path.name.split('.')[-1]
        handler_stack = Cache.HANDLER_STACK
        if self._custom_handler is not None:
            handler_stack=[self._custom_handler]
        for handler in handler_stack:
            if handler.get_extension()==extension:
                return handler.read(path)
        raise ValueError(f'No handler was found for extension {extension}, path {path}. Custom handler is {self._custom_handler}')

    def _make_cache(self, data) -> None:
        path = self._get_path_prefix()
        handler_stack = Cache.HANDLER_STACK
        if self._custom_handler is not None:
            handler_stack=[self._custom_handler]
        for handler in handler_stack:
            if handler.accepts_object(data):
                path = Path(str(path)+'.'+handler.get_extension())
                handler.write(data, path)
                return
        raise ValueError(f'No handler was found for data type {type(data)}. Custom handler is {self._custom_handler}')

    def get_data(self) -> TData:
        if self._cache_mode == Cache.Mode.Use or self._cache_mode == 'use':
            return self._from_cache()
        elif self._cache_mode == Cache.Mode.Remake or self._cache_mode == 'remake':
            data = self._source.get_data()
            self._make_cache(data)
            return data
        elif self._cache_mode == Cache.Mode.No or self._cache_mode == 'no':
            return self._source.get_data()
        else:
            path = self._get_path()
            if path is None:
                data = self._source.get_data()
                self._make_cache(data)
                return data
            else:
                return self._from_cache()


    HANDLER_STACK = (
        PandasParquetHandler(),
        DataBundleHandler(),
        DefaultPickleHandler()
    )

    DEFAULT_FOLDER = Loc.data_cache_path/'caches'

    DEFAULT_CACHE_MODE = 'default'

    @staticmethod
    def source(source, *args, **kwargs) -> 'Cache':
        if isinstance(source, IAccessor):
            return Cache(source)
        else:
            return Cache(FunctionalDataAccessor(source, *args, **kwargs))



    @staticmethod
    def default_name(prefix, *args, **kwargs):
        name = prefix
        suffix = []
        for arg in args:
            suffix.append(str(arg))
        for key, value in kwargs.items():
            suffix.append(str(key))
            suffix.append(str(value))
        suffix='_'.join(suffix)
        if len(suffix)>100:
            suffix = md5(suffix.encode('utf-8')).hexdigest()
        result = name+'_'+suffix
        result = result.replace('/','_').replace('\\', '_').replace('?', '_')
        return result