import typing

from openff.nagl._base.base import ImmutableModel


class OptimizerConfig(ImmutableModel):
    optimizer: typing.Literal["Adam"]
    learning_rate: float
