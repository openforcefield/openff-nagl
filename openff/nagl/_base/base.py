import enum
import hashlib
import inspect
import pathlib
import json
import yaml
from typing import Any, ClassVar, Dict, List, Optional, Type, no_type_check

import numpy as np
from openff.units import unit

from ..utils._utils import round_floats

try:
    from pydantic.v1 import BaseModel
    from pydantic.v1.errors import DictError
except ImportError:
    from pydantic import BaseModel
    from pydantic.errors import DictError

class MutableModel(BaseModel):
    """
    Base class that all classes should subclass.
    """

    class Config:
        validate_all = True
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
            tuple: list,
            set: list,
            unit.Quantity: lambda x: x.to_tuple(),
            enum.Enum: lambda x: x.name,
            pathlib.Path: str,
        }

    _hash_fields: ClassVar[Optional[List[str]]] = None
    _float_fields: ClassVar[List[str]] = []
    _float_decimals: ClassVar[int] = 8
    _hash_int: Optional[int] = None
    _hash_str: Optional[str] = None

    def __init__(self, *args, **kwargs):
        self.__pre_init__(*args, **kwargs)
        super(MutableModel, self).__init__(*args, **kwargs)
        self.__post_init__(*args, **kwargs)

    def __pre_init__(self, *args, **kwargs):
        pass

    def __post_init__(self, *args, **kwargs):
        pass

    # def __eq__(self, other):
    #     return hash(self) == hash(other)


    def to_json(self):
        return self.json(
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )

    @classmethod
    def from_json(cls, string_or_file):
        try:
            with open(string_or_file, "r") as f:
                string_or_file = f.read()
        except (OSError, FileNotFoundError):
            pass
        return cls.parse_raw(string_or_file)

    def to_yaml(self, filename):
        data = json.loads(self.json())
        with open(filename, "w") as f:
            yaml.dump(data, f)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**data)

class ImmutableModel(MutableModel):
    class Config(MutableModel.Config):
        allow_mutation = False
