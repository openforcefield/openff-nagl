"""
Location of data files for tests
================================

Use as ::

    from openff.nagl.tests.data.files import *

"""

__all__ = [
    # "EXAMPLE_MODEL_CONFIG",
    # "MODEL_CONFIG_V7",
    # "EXAMPLE_AM1BCC_MODEL_STATE_DICT",
    "EXAMPLE_AM1BCC_MODEL",
    "EXAMPLE_UNFEATURIZED_PARQUET_DATASET",
    "EXAMPLE_FEATURIZED_PARQUET_DATASET",
    "EXAMPLE_UNFEATURIZED_PARQUET_DATASET_SHORT",
    "EXAMPLE_FEATURIZED_PARQUET_DATASET_SHORT",
    "EXAMPLE_TRAINING_CONFIG",
    "EXAMPLE_TRAINING_CONFIG_LAZY",
    "EXAMPLE_FEATURIZED_LAZY_DATA",
    "EXAMPLE_FEATURIZED_LAZY_DATA_SHORT"
]

import importlib_resources

data_directory = importlib_resources.files("openff.nagl") / "tests" / "data"

EXAMPLE_AM1BCC_MODEL = data_directory / "example_am1bcc_model.pt"
EXAMPLE_UNFEATURIZED_PARQUET_DATASET = data_directory / "example-data-labelled-unfeaturized"
EXAMPLE_FEATURIZED_PARQUET_DATASET = data_directory / "example-data-labelled-featurized"
EXAMPLE_UNFEATURIZED_PARQUET_DATASET_SHORT = data_directory / "example-data-labelled-unfeaturized-short"
EXAMPLE_FEATURIZED_PARQUET_DATASET_SHORT = data_directory / "example-data-labelled-featurized-short"
EXAMPLE_TRAINING_CONFIG = data_directory / "example_training_config.yaml"
EXAMPLE_TRAINING_CONFIG_LAZY = data_directory / "example_training_config_lazy.yaml"

EXAMPLE_FEATURIZED_LAZY_DATA = data_directory / "cbe6f394311f594a9df33d7580e8b8478f0aef5b505f16f8b2f6af721a14e30d.arrow"
EXAMPLE_FEATURIZED_LAZY_DATA_SHORT = data_directory / "b6713a9ba87e89cb53d264256664bb9e4e4a5831f0c7660808e9f44a9f832ab5.arrow"
