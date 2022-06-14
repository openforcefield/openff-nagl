from typing import List, Union, TYPE_CHECKING, Tuple

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


