"""
Location of data files for tests
================================

Use as ::

    from openff.nagl.tests.data.files import *

"""

__all__ = [
    "EXAMPLE_MODEL_CONFIG",
    "MODEL_CONFIG_V7",
    "EXAMPLE_AM1BCC_MODEL_STATE_DICT",
    "EXAMPLE_AM1BCC_MODEL",
]

from pkg_resources import resource_filename

EXAMPLE_MODEL_CONFIG = resource_filename(__name__, "example_model_config.yaml")
MODEL_CONFIG_V7 = resource_filename(__name__, "model_config_v7.yaml")
EXAMPLE_AM1BCC_MODEL_STATE_DICT = resource_filename(__name__, "example_am1bcc_model_state_dict.pt")
EXAMPLE_AM1BCC_MODEL = resource_filename(__name__, "example_am1bcc_model.pt")
