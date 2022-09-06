import enum
from typing import Callable

import torch


class ActivationFunction(enum.Enum):
    Identity = torch.nn.Identity
    Tanh = torch.nn.Tanh
    ReLU = torch.nn.ReLU
    LeakyReLU = torch.nn.LeakyReLU
    ELU = torch.nn.ELU

    @classmethod
    def _lowercase(cls):
        return {
            name.lower(): value
            for name, value in cls.__members__.items()
        }

    @classmethod
    def get(cls, name: str) -> "ActivationFunction":
        if isinstance(name, cls):
            return name
        if isinstance(name, str):
            try:
                return cls[name]
            except KeyError:
                return cls._lowercase()[name.lower()]
        return cls(name)

    @classmethod
    def get_value(cls, name: str) -> Callable[[torch.tensor], torch.Tensor]:
        try:
            return cls.get(name).value()
        except ValueError:
            return name

