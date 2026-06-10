"""
A toolkit for the generation of neural network models for predicting molecule
properties.
"""

from importlib.metadata import version

__version__ = version("openff.nagl")

from openff.nagl.nn._models import GNNModel

__all__ = [
    "GNNModel",
]
