from typing import TypeVar

import torch

T = TypeVar("T")


def one_hot_encode[T](value: T, categories: list[T]) -> torch.tensor:
    """
    One-hot encode a value.
    """
    tensor = torch.zeros((1, len(categories)), dtype=torch.int64)
    tensor[0, categories.index(value)] = 1
    return tensor
