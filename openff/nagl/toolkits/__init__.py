"Extensions of the OpenFF Toolkit ToolkitWrappers for use with NAGL."

from typing import Union

from ._base import (
    NAGLToolkitWrapperBase,
    NAGLToolkitWrapperMeta,
    ToolkitWrapperType,
)
from .openeye import NAGLOpenEyeToolkitWrapper
from .rdkit import NAGLRDKitToolkitWrapper
from .registry import NAGLToolkitRegistry

NAGL_TOOLKIT_REGISTRY = NAGLToolkitRegistry(
    [NAGLOpenEyeToolkitWrapper, NAGLRDKitToolkitWrapper], exception_if_unavailable=False
)


ToolkitRegistryType = Union[NAGLToolkitRegistry, ToolkitWrapperType]

__all__ = [
    "NAGLToolkitWrapperBase",
    "NAGLToolkitWrapperMeta",
    "NAGLOpenEyeToolkitWrapper",
    "NAGLRDKitToolkitWrapper",
    "NAGL_TOOLKIT_REGISTRY",
]
