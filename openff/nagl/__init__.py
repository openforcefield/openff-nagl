"""
A toolkit for the generation of neural network models for predicting molecule
properties.
"""

from openff.nagl.nn._models import GNNModel

__all__ = [
    "GNNModel",
]

# Handle versioneer
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
