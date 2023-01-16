# Toolkit wrappers

NAGL toolkit wrappers are subclasses of the OpenFF Toolkit [`ToolkitWrapper`] class with additional functionality. They are intended to be used in the same way. As they inherit from the Toolkit, all functionality of the OpenFF Toolkit is included. Therefore when working with NAGL, it is likely easier to use the NAGL ToolkitWrappers rather than the OpenFF Toolkit.

One difference is that a [`NAGLToolkitRegistry`] can be created in three ways:

```python
from openff.nagl.utils.toolkits import (
    NAGLToolkitRegistry,
    NAGLOpenEyeToolkitWrapper,
    NAGLRDKitToolkitWrapper
)

# Create a registry with classes
registry = NAGLToolkitRegistry(
    toolkit_wrappers=[NAGLOpenEyeToolkitWrapper, NAGLRDKitToolkitWrapper]
)

# Create a registry with instances
registry = NAGLToolkitRegistry(
    toolkit_wrappers=[NAGLOpenEyeToolkitWrapper(), NAGLRDKitToolkitWrapper()]
)

# Create a registry with strings
registry = NAGLToolkitRegistry(
    toolkit_wrappers=["openeye", "rdkit"]
)
```

Similarly, any method that takes a `toolkit_registry` argument can take a class, instance, or the name of a toolkit wrapper.

[`NAGLToolkitRegistry`]: openff.nagl.toolkits.registry.NAGLToolkitRegistry
[`ToolkitWrapper`]: openff.toolkit.utils.toolkits.ToolkitWrapper