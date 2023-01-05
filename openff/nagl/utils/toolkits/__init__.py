from typing import Union

from .base import NAGLToolkitWrapperBase, NAGLToolkitWrapperMeta, ToolkitWrapperType
from .registry import NAGLToolkitRegistry
from .openeye import NAGLOpenEyeToolkitWrapper
from .rdkit import NAGLRDKitToolkitWrapper

NAGL_TOOLKIT_REGISTRY = NAGLToolkitRegistry([NAGLOpenEyeToolkitWrapper, NAGLRDKitToolkitWrapper])
ToolkitRegistryType = Union[NAGLToolkitRegistry, ToolkitWrapperType]