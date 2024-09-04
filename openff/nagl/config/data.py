"""
Config classes for training data.
"""

import pathlib
import typing

from openff.nagl.training.loss import (
    TargetType,

)
from openff.nagl._base.base import ImmutableModel
from openff.nagl.utils._types import FromYamlMixin

try:
    from pydantic.v1 import Field
except ImportError:
    from pydantic import Field

DiscriminatedTargetType = typing.Annotated[TargetType, Field(discriminator="name")]

class DatasetConfig(ImmutableModel, FromYamlMixin):
    """
    A config class for a single dataset. Datasets can be combined from
    multiple data `sources` and can be used for training or validation.
    Multiple targets can be defined that read different columns from the training
    sets. The required columns must be present in all `sources`.
    """
    sources: typing.Optional[list[str]] = Field(
        None,
        description=(
            "Paths to data sources. "
            "The data should be formatted to be readable as PyArrow dataset. "
            "Sources can be a single file or a directory of files."
        )
    )
    targets: list[DiscriminatedTargetType] = Field(
        description="Targets to train or evaluate against",
    )
    batch_size: typing.Optional[int] = Field(
        None,
        description="Batch size to use"
    )
    use_cached_data: bool = Field(
        default=False,
        description="Whether to use cached data",
    )
    cache_directory: typing.Optional[pathlib.Path] = Field(
        default=None,
        description="Directory to read cached data from, or cache data in",
    )
    lazy_loading: bool = Field(
        default=False,
        description="Whether to lazily load data",
    )
    n_processes: int = Field(
        default=0,
        description="Number of processes to use for loading data",
    )

    def get_required_target_columns(self) -> list[str]:
        """Get all required columns from the datasets for the targets"""
        columns = set()
        for target in self.targets:
            columns |= set(target.get_required_columns())
        return sorted(columns)


class DataConfig(ImmutableModel, FromYamlMixin):
    """
    A config class for setting up training, validation, and test datasets.
    """
    training: DatasetConfig = Field(description="Training dataset")
    validation: typing.Optional[DatasetConfig] = Field(
        default=None,
        description="Validation dataset",
    )
    test: typing.Optional[DatasetConfig] = Field(
        default=None,
        description="Test dataset",
    )
    
    def get_required_target_columns(self) -> list[str]:
        """Get all required columns from the datasets for the targets"""
        columns = set()
        columns |= set(self.training.get_required_target_columns())
        if self.validation is not None:
            columns |= set(self.validation.get_required_target_columns())
        if self.test is not None:
            columns |= set(self.test.get_required_target_columns())
        return sorted(columns)