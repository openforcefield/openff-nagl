import hashlib
import inspect
from typing import Any, Dict, Optional, Type, ClassVar, List, no_type_check, Tuple


from pydantic import BaseModel, validator
from pydantic.errors import DictError
import numpy as np
import networkx as nx
from openff.units import unit
from rdkit import Chem

from ..utils.utils import round_floats


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
            unit.Quantity: lambda x: x.to_tuple(),
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

    @classmethod
    def _get_properties(cls) -> Dict[str, property]:
        return dict(
            inspect.getmembers(
                cls,
                predicate=lambda x: isinstance(x, property)
            )
        )

    @no_type_check
    def __setattr__(self, attr, value):
        try:
            super().__setattr__(attr, value)
        except ValueError as e:
            properties = self._get_properties()
            if attr in properties:
                if properties[attr].fset is not None:
                    return properties[attr].fset(self, value)
            raise e

    def _clsname(self):
        return type(self).__name__

    def _clear_hash(self):
        self._hash_int = None
        self._hash_str = None

    def _set_attr(self, attrname, value):
        self.__dict__[attrname] = value
        self._clear_hash()

    def __hash__(self):
        if self._hash_int is None:
            mash = self.get_hash()
            self._hash_int = int(mash, 16)
        return self._hash_int

    def get_hash(self) -> str:
        """Returns string hash of the object"""
        if self._hash_str is None:
            dumped = self.dumps(decimals=self._float_decimals)
            mash = hashlib.sha1()
            mash.update(dumped.encode("utf-8"))
            self._hash_str = mash.hexdigest()
        return self._hash_str

    def __eq__(self, other):
        return hash(self) == hash(other)

    def hash_dict(self) -> Dict[str, Any]:
        """Create dictionary from hash fields and sort alphabetically"""
        if self._hash_fields:
            hashdct = self.dict(include=set(self._hash_fields))
        else:
            hashdct = self.dict()
        data = {k: hashdct[k] for k in sorted(hashdct)}
        return data

    def dumps(self, decimals: Optional[int] = None):
        """Serialize object to a JSON formatted string

        Unlike json(), this method only includes hashable fields,
        sorts them alphabetically, and optionally rounds floats.
        """
        data = self.hash_dict()
        dump = self.__config__.json_dumps
        if decimals is not None:
            for field in self._float_fields:
                if field in data:
                    data[field] = round_floats(data[field],
                                               decimals=decimals)
            with np.printoptions(precision=16):
                return dump(data, default=self.__json_encoder__)
        return dump(data, default=self.__json_encoder__)

    def _round(self, obj):
        return round_floats(obj, decimals=self._float_decimals)

    def to_json(self):
        return self.json(sort_keys=True, indent=2, separators=(",", ": "),)

    @classmethod
    def _from_dict(cls, **kwargs):
        dct = {
            k: kwargs[k]
            for k in kwargs
            if k in cls.__fields__
        }
        return cls(**dct)

    @classmethod
    def validate(cls: Type['Model'], value: Any) -> 'Model':
        if isinstance(value, dict):
            return cls(**value)
        elif isinstance(value, cls):
            return value
        elif cls.__config__.orm_mode:
            return cls.from_orm(value)
        elif cls.__custom_root_type__:
            return cls.parse_obj(value)
        else:
            try:
                value_as_dict = dict(value)
            except (TypeError, ValueError) as e:
                raise DictError() from e
            return cls(**value_as_dict)

    @classmethod
    def from_json(cls, string_or_file):
        try:
            with open(string_or_file, "r") as f:
                string_or_file = f.read()
        except (OSError, FileNotFoundError):
            pass
        return cls.parse_raw(string_or_file)

    def copy(self, *, include=None, exclude=None, update=None, deep: bool = False):
        obj = super().copy(include=include, exclude=exclude, update=update, deep=deep)
        obj.__post_init__()
        return obj

    def _replace_from_mapping(self, attr_name, mapping_values={}):
        current_value = getattr(self, attr_name)
        if current_value in mapping_values:
            self._set_attr(attr_name, mapping_values[current_value])


class ImmutableModel(MutableModel):
    class Config(MutableModel.Config):
        allow_mutation = False

