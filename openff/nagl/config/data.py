import typing

from pydantic import Field

from openff.nagl.nn._loss import _BaseTarget
from openff.nagl._base.base import ImmutableModel



class Dataset(ImmutableModel):
    sources: typing.Optional[typing.List[str]] = Field(
        None,
        description="Paths to data"
    )
    targets: typing.List[_BaseTarget] = Field(
        description="Targets to train or evaluate against"
    )
    batch_size: typing.Optional[int] = Field(
        None,
        description="Batch size to use"
    )


class DataConfig(ImmutableModel):
    training: Dataset = Field(description="Training dataset")
    validation: typing.Optional[Dataset] = Field(
        None,
        description="Validation dataset",
    )
    test: typing.Optional[Dataset] = Field(
        None,
        description="Test dataset",
    )