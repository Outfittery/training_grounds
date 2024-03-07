from typing import *
from yo_fluq_ds import FileIO
from enum import Enum
from pathlib import Path

class CacheMode(Enum):
    """
    Contains the modes for the cacheable data source.
    `No` means the cache will not be used at all.
    `Default` means the cache will be read if available, and created otherwise
    `Use` means the cache will be read if available, but if not, the exception is going to be raised
    `Remake` means the cache will be recreated regardless of its existence
    """
    No = 0
    Default = 1
    Use = 2
    Remake = 3


    @staticmethod
    def parse(value: Union[None, str, 'CacheMode']) -> 'CacheMode':
        if value is None:
            return CacheMode.Default
        elif isinstance(value, CacheMode):
            return value
        elif value == 'default':
            return CacheMode.Default
        elif value == 'no':
            return CacheMode.No
        elif value == 'use':
            return CacheMode.Use
        elif value == 'remake':
            return CacheMode.Remake
        raise ValueError(f'Cannot recognize value {value}')

