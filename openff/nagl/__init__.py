"""
A toolkit for the generation of neural network models for predicting molecule
properties.
"""

# Set __version__ first so that openff.toolkit's NAGLToolkitWrapper.__init__
# can read it even if openff.nagl is only partially initialized. Without this,
# importing any openff.nagl.toolkits submodule triggers a circular import:
# toolkits._base → openff.toolkit.utils → GLOBAL_TOOLKIT_REGISTRY creation
# → NAGLToolkitWrapper() → `from openff.nagl import __version__` → ImportError.
from . import _version
__version__ = _version.get_versions()['version']

from openff.nagl.nn._models import GNNModel

__all__ = [
    "GNNModel",
]
