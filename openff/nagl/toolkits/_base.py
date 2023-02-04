import importlib
from typing import Union

from openff.nagl._base.metaregistry import create_registry_metaclass
from openff.toolkit.utils.base_wrapper import ToolkitWrapper


class NAGLToolkitWrapperMeta(create_registry_metaclass("name", ignore_case=True)):
    pass


class NAGLToolkitWrapperBase(ToolkitWrapper, metaclass=NAGLToolkitWrapperMeta):
    pass


ToolkitWrapperType = Union[NAGLToolkitWrapperMeta, NAGLToolkitWrapperBase, str]
