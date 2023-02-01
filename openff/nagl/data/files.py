"""
Location of data files
======================

Use as ::

    from openff.nagl.data.files import *

"""

__all__ = ["MOLECULE_NORMALIZATION_REACTIONS", "EXAMPLE_MODEL_CONFIG", "EXAMPLE_AM1BCC_MODEL"]

from pkg_resources import resource_filename

MOLECULE_NORMALIZATION_REACTIONS = resource_filename(__name__, "normalizations.json")
EXAMPLE_MODEL_CONFIG = resource_filename(__name__, "example_model_config.yaml")
MODEL_CONFIG_V7 = resource_filename(__name__, "model_config_v7.yaml")

EXAMPLE_AM1BCC_MODEL = resource_filename(__name__, "example_am1bcc_model.pt")
# EXAMPLE_AM1BCC_MODEL_STATE_DICT = resource_filename(__name__, "example_am1bcc_model.pt")