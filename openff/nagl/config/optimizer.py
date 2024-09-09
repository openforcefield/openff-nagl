"""Config classes for defining optimizers"""

import typing

from openff.nagl._base.base import ImmutableModel


class OptimizerConfig(ImmutableModel):
    """The configuration for the optimizer to use during training"""
    optimizer: typing.Literal["Adam"]
    learning_rate: float