from typing import Union

from openff.toolkit.utils.base_wrapper import ToolkitWrapper

from openff.nagl._base.metaregistry import create_registry_metaclass


class NAGLToolkitWrapperMeta(create_registry_metaclass("name", ignore_case=True)):
    pass


class NAGLToolkitWrapperBase(ToolkitWrapper, metaclass=NAGLToolkitWrapperMeta):
    pass


ToolkitWrapperType = Union[NAGLToolkitWrapperMeta, NAGLToolkitWrapperBase, str]
