import numpy as np
import pytest
import torch
from openff.toolkit.topology.molecule import unit as offunit

from openff.nagl.utils._utils import (
    as_iterable,
    assert_same_lengths,
    is_iterable,
)


@pytest.mark.parametrize(
    "obj, expected",
    [
        ("asdf", False),
        (None, False),
        (1, False),
        (1.0, False),
        ([1], True),
        ({3, 4}, True),
        (dict(), True),
        (np.arange(3), True),
        (np.arange(3) * offunit.angstrom, True),
        (torch.tensor([1, 2, 3]), True),
    ],
)
def test_is_iterable(obj, expected):
    assert is_iterable(obj) == expected


@pytest.mark.parametrize(
    "obj, expected",
    [
        ("asdf", ["asdf"]),
        (None, [None]),
        (1, [1]),
        (1.0, [1.0]),
        ([1], [1]),
        ({3, 4}, {3, 4}),
        (dict(), {}),
    ],
)
def test_as_iterable(obj, expected):
    assert as_iterable(obj) == expected


def test_assert_same_lengths():
    assert_same_lengths([1, 2, 3], ["a", "b", "c"])


def test_assert_same_lengths_incorrect_types():
    with pytest.raises(TypeError, match="must be iterable"):
        assert_same_lengths([1, 2, 3], None)


def test_assert_same_lengths_incorrect_lengths():
    with pytest.raises(AssertionError, match="must have the same length"):
        assert_same_lengths([1, 2, 3], [1, 2])
