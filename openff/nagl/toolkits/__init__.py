"Extensions of the OpenFF Toolkit ToolkitWrappers for use with NAGL."

from .registry import NAGLToolkitRegistry
from .openeye import NAGLOpenEyeToolkitWrapper
from .rdkit import NAGLRDKitToolkitWrapper

NAGL_TOOLKIT_REGISTRY = NAGLToolkitRegistry(
    [NAGLOpenEyeToolkitWrapper, NAGLRDKitToolkitWrapper], exception_if_unavailable=False
)


__all__ = [
    "NAGLOpenEyeToolkitWrapper",
    "NAGLRDKitToolkitWrapper",
    "NAGL_TOOLKIT_REGISTRY",
]
