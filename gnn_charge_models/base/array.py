from typing import Any

import numpy as np


class ArrayMeta(type):
    def __getitem__(cls, T):
        return type("Array", (Array,), {"__dtype__": T})


class Array(np.ndarray, metaclass=ArrayMeta):
    """A typeable numpy array"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        dtype = getattr(cls, "__dtype__", Any)
        if dtype is Any:
            dtype = None
        return np.asanyarray(val, dtype=dtype)
