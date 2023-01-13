"""
OpenFF NAGL
A toolkit for the generation of neural network models for predicting molecule properties.
"""

from openff.nagl.nn._models import GNNModel

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
