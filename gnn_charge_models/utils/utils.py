from typing import List, Union, TYPE_CHECKING, Any, Iterable, Optional


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


def transform_coordinates(
    coordinates: np.ndarray,
    scale: float = 1.0,
    translate: Optional[float] = 0.0,
    rotate: Optional[float] = 0.0,
) -> np.ndarray:
    """
    Transform the coordinates by a scale, translation, and rotation.

    Parameters
    ----------
    coordinates : np.ndarray
        The coordinates to transform.
    scale : float, optional
        The scale to apply to the coordinates.
        If None, a random scale will be generated
        using numpy.random.random
    translate : float, optional
        The translation to apply to the coordinates.
        If None, a random translation will be generated
        using numpy.random.random
    rotate : float, optional
        The rotation to apply to the coordinates.
        If None, a random angle will be generated
        using numpy.random.random
    """
    if rotate is None:
        rotate = np.random.random()
    if translate is None:
        translate = np.random.random()
    if scale is None:
        scale = np.random.random()

    cos_theta, sin_theta = np.cos(rotate), np.sin(rotate)
    rot_matrix = np.array([
        [cos_theta, 0.0, -sin_theta],
        [0.0, 1.0, 0.0],
        [sin_theta, 0.0, cos_theta],
    ])

    coordinates = coordinates.reshape((-1, 3))
    centered = coordinates - coordinates.mean(axis=0)
    scaled = centered * scale
    rotated = scaled @ rot_matrix
    translated = rotated + translate

    return translated
