import importlib
from typing import Union

from openff.nagl.base.metaregistry import create_registry_metaclass
from openff.toolkit.utils import exceptions as toolkit_exceptions
from openff.toolkit.utils.exceptions import ToolkitUnavailableException

class ToolkitWrapperMeta(create_registry_metaclass("name", ignore_case=True)):
    pass

class ToolkitWrapperBase(metaclass=ToolkitWrapperMeta):

    unavailable_message: str = (
        "You may need to install it or obtain a license."
    )

    def __init__(self):
        if not self.is_available():
            raise toolkit_exceptions.ToolkitUnavailableException(
                f"Toolkit {self.name} is not available. {self.unavailable_message}"
            )
        
        self._toolkit_version = importlib.import_module(self.name).__version__
    
    def is_available(self):
        return False


ToolkitWrapperType = Union[ToolkitWrapperMeta, ToolkitWrapperBase, str]