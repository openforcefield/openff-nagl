"""
A toolkit for the generation of neural network models for predicting molecule
properties.
"""

from openff.nagl.nn._models import GNNModel

__all__ = [
    "GNNModel",
]

# Handle versioneer
from . import _version

__version__ = _version.get_versions()["version"]
