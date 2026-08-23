import enum
import pathlib
import json
import yaml

import numpy as np
from openff.units import unit


from pydantic import BaseModel, model_serializer, ConfigDict

def _encode_values(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, enum.Enum):
        return obj.name.lower()
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, (tuple, set)):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _encode_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode_values(i) for i in obj]
    return obj

class MutableModel(BaseModel):
    """
    Base class that all classes should subclass.
    """

    model_config = ConfigDict(
        validate_default=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        return _encode_values(data)

    def __init__(self, *args, **kwargs):
        self.__pre_init__(*args, **kwargs)
        super(MutableModel, self).__init__(*args, **kwargs)
        self.__post_init__(*args, **kwargs)

    def __pre_init__(self, *args, **kwargs):
        pass

    def __post_init__(self, *args, **kwargs):
        pass

    def model_dump_json(self, **kwargs):
        return json.dumps(
            self.model_dump(**kwargs),
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
        data = json.loads(self.model_dump_json())
        with open(filename, "w") as f:
            yaml.dump(data, f)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**data)

class ImmutableModel(MutableModel):
    # other options are **merged** with parent's config_dict
    model_config = ConfigDict(frozen=True)
