"""
Location of data files
======================

Use as ::

    from openff.nagl.data.files import *

"""

__all__ = ["MOLECULE_NORMALIZATION_REACTIONS", "EXAMPLE_MODEL_CONFIG"]

from pkg_resources import resource_filename

MOLECULE_NORMALIZATION_REACTIONS = resource_filename(__name__, "normalizations.json")
EXAMPLE_MODEL_CONFIG = resource_filename(__name__, "example_model_config.yaml")
