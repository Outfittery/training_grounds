from typing import *

import pickle
import os

from yo_fluq_ds import Query
from pathlib import Path

from ..._common.locations import Loc



class HackedUnpicker(pickle.Unpickler):
    """
    The class to read pickled files, whose classes changed the name
    """
    def __init__(self, file_obj, from_module, to_module, additional_replacements = None):
        super(HackedUnpicker, self).__init__(file_obj)
        self.from_module = from_module
        self.to_module = to_module
        self.additional_replacements = additional_replacements

    def find_class(self, module: str, name: str) -> str:
        """
        Overriden. Replaces the module for training ground with another one.
        """
        renamed_module = module

        prefix = self.from_module
        if module.startswith(prefix):
            renamed_module = self.to_module + module[len(prefix):]

        if self.additional_replacements is not None:
            for key, value in self.additional_replacements.items():
                if renamed_module.startswith(key):
                    renamed_module = value+renamed_module[len(key):]
                    break


        return super(HackedUnpicker, self).find_class(renamed_module, name)


class EntryPoint:
    """
    This class describes the TG-package.
    """
    def __init__(self,
                 module_name: str,
                 module_version: str,
                 python_module_name: str,
                 tg_module_name: str,
                 original_tg_module_name: str,
                 resources_location: Path
                 ):
        """
        You don't have to create the instances of this class yourself. The instances are obtained from the installed TG-packages
        Args:
            module_name: the name of the module, which is typically the name of the project + milestone version
            module_version: the version of the package.
            python_module_name: the name of the top-level module, usually the same as ``module-name``
            tg_module_name: currently always ``tg``
            original_tg_module_name: currently always ``tg``
            resources_location: the name of resource folder inside the package
        """
        self.module_name = module_name
        self.module_version = module_version
        self.tg_module_name = tg_module_name
        self.python_module_name = python_module_name
        self.original_tg_module_name = original_tg_module_name
        self.resources_location = str(resources_location)

    def get_properties(self):
        """
        Returns all the internal properties: module's name, version, etc.
        """
        return self.__dict__


    def get_resources(self) -> List[str]:
        """
        Returns list of resources' names

        """
        return Query.folder(self.resources_location).select(lambda z: z.name).to_list()

    def load_resource(self, resource_name: str) -> Any:
        """
        Loads the resource by its name
        """
        with open(os.path.join(self.resources_location,resource_name), 'rb') as file_obj:
            return HackedUnpicker(file_obj,self.original_tg_module_name, self.tg_module_name).load()


    def load_file_in_local_tg(self, filename: str, local_tg: Optional[str] = None):
        """
        Loads the file, translating its schema from this package to the "normal" TG package (``tg`` in the current environment)

        """
        if local_tg is None:
            local_tg = Loc.tg_name
        with open(filename,'rb') as file_obj:
            return HackedUnpicker(file_obj,self.tg_module_name,local_tg).load()






