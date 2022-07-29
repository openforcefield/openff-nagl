from typing import Any
from openff.units import unit as off_unit

from pydantic.class_validators import prep_validators


class QuantityMeta(type):
    def __getitem__(cls, unit):
        dtype = None
        if isinstance(unit, tuple):
            unit, dtype = unit
        return type("Quantity", (Quantity,), {"__unit__": unit, "__dtype__": dtype})


class Quantity(off_unit.Quantity, metaclass=QuantityMeta):
    @classmethod
    def __get_validators__(cls):
        yield from (
            cls.validate_type,
            cls.validate_unit,
        )

    @classmethod
    def validate_type(cls, val):
        dtype = getattr(cls, "__dtype__", Any)
        if dtype is Any:
            dtype = None
        get_validators = getattr(val, "__get_validators__", None)
        if get_validators is not None:
            validators = prep_validators(get_validators())
            for validator in validators:
                val = validator(val)

        elif not isinstance(val, dtype):
            raise TypeError(f"{val} is not of type {dtype}")

        return val

    @classmethod
    def validate_unit(cls, val):
        unit = getattr(cls, "__unit__", None)
        if unit is None:
            return val
        if isinstance(val, off_unit.Quantity):
            assert val.is_compatible_with(unit)
        else:
            val = val * unit
        return val
