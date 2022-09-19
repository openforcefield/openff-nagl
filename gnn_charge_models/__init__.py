"""
GNN Charge Models
A short description of the project.
"""

# Handle versioneer
from ._version import get_versions
# Add imports here
from .gnn_charge_models import *

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
