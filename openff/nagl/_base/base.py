import enum
import pathlib
import json
import yaml

import numpy as np
from openff.units import unit


try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

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

    def __init__(self, *args, **kwargs):
        self.__pre_init__(*args, **kwargs)
        super(MutableModel, self).__init__(*args, **kwargs)
        self.__post_init__(*args, **kwargs)

    def __pre_init__(self, *args, **kwargs):
        pass

    def __post_init__(self, *args, **kwargs):
        pass

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
        try:
            validator = cls.model_validate_json
        except AttributeError:
            validator = cls.parse_raw
        return validator(string_or_file)

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
