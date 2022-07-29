from typing import List, Union, TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule

FloatArrayLike = Union[List, np.ndarray, float]


def round_floats(
    obj: FloatArrayLike,
    decimals: int = 8,
) -> FloatArrayLike:
    rounded = np.around(obj, decimals)
    threshold = 5 ** (1 - decimals)
    if isinstance(rounded, np.ndarray):
        rounded[np.abs(rounded) < threshold] = 0.0
    elif np.abs(rounded) < threshold:
        rounded = 0.0
    return rounded


def assert_same_lengths(*values):
    try:
        lengths = [len(value) for value in values]
    except TypeError:
        raise TypeError("All values must be iterable.") from None
    err = f"All values must have the same length: {lengths}"
    assert len(set(lengths)) == 1, err


def is_iterable(obj: Any) -> bool:
    from collections.abc import Iterable
    return isinstance(obj, Iterable) and not isinstance(obj, str)

def as_iterable(obj: Any) -> Iterable:
    if not is_iterable(obj):
        return [obj]
    return obj
