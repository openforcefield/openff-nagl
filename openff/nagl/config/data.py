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
    sources: typing.Optional[typing.List[str]] = Field(
        None,
        description="Paths to data"
    )
    targets: typing.List[DiscriminatedTargetType] = Field(
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
    determine_cache_file_from_paths: bool = Field(
        default=False,
        description=(
            "The cache filename is determined in part from the sources. "
            "If this is True, the cache filename will be determined from the paths. "
            "That can be dangerous if datasets get modified or updated, but it is "
            "much faster if the datasets are large and the paths are stable. "
            "If this is False, the cache filename will be determined from the "
            "contents of the files. That is safer, but slower."
        ),
    )
    lazy_loading: bool = Field(
        default=False,
        description="Whether to lazily load data",
    )
    n_processes: int = Field(
        default=0,
        description="Number of processes to use for loading data",
    )

    def get_required_target_columns(self):
        columns = set()
        for target in self.targets:
            columns |= set(target.get_required_columns())
        return sorted(columns)


class DataConfig(ImmutableModel, FromYamlMixin):
    training: DatasetConfig = Field(description="Training dataset")
    validation: typing.Optional[DatasetConfig] = Field(
        default=None,
        description="Validation dataset",
    )
    test: typing.Optional[DatasetConfig] = Field(
        default=None,
        description="Test dataset",
    )
    
    def get_required_target_columns(self):
        columns = set()
        columns |= set(self.training.get_required_target_columns())
        if self.validation is not None:
            columns |= set(self.validation.get_required_target_columns())
        if self.test is not None:
            columns |= set(self.test.get_required_target_columns())
        return sorted(columns)