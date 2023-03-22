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

import importlib_resources

data_directory = importlib_resources.files("openff.nagl") / "tests" / "data"

EXAMPLE_MODEL_CONFIG = data_directory / "example_model_config.yaml"
MODEL_CONFIG_V7 = data_directory / "model_config_v7.yaml"
EXAMPLE_AM1BCC_MODEL_STATE_DICT = data_directory / "example_am1bcc_model_state_dict.pt"
EXAMPLE_AM1BCC_MODEL = data_directory / "example_am1bcc_model.pt"
