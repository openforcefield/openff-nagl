from typing import List, TypeVar

import torch

__all__ = [
    "one_hot_encode",
]


T = TypeVar("T")


def one_hot_encode(value: T, categories: List[T]) -> torch.tensor:
    """
    One-hot encode a value.
    """
    tensor = torch.zeros((1, len(categories)), dtype=torch.int64)
    tensor[0, categories.index(value)] = 1
    return tensor
