"""
Location of data files
======================

Use as ::

    from gnn_charge_models.data.files import *

"""

__all__ = [
    "MOLECULE_NORMALIZATION_REACTIONS",
]

from pkg_resources import resource_filename

MOLECULE_NORMALIZATION_REACTIONS = resource_filename(__name__, "normalizations.json")
