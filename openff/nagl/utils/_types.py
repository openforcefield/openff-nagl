import enum
import pathlib
from typing import Union, Tuple, NamedTuple, Dict, Literal

Pathlike = Union[str, pathlib.Path]


class HybridizationType(enum.Enum):
    OTHER = "other"
    SP = "sp"
    SP2 = "sp2"
    SP3 = "sp3"
    SP3D = "sp3d"
    SP3D2 = "sp3d2"


class ResonanceAtomType(enum.Enum):
    Acceptor = "A"
    Donor = "D"


class ResonanceType:

    class Key(NamedTuple):
        """A convenient data structure for storing information used to recognize a possible
        resonance atom type by."""

        atomic_number: Literal[8, 16, 7]
        formal_charge: int
        bond_orders: Tuple[int, ...]

    class Value(NamedTuple):
        """A convenient data structure for storing information about a possible resonance
        atom type in."""

        type: ResonanceAtomType
        energy: float
        id: int
        conjugate_id: int

        def get_conjugate_key(self):
            return ResonanceType._resonance_keys_by_id[self.conjugate_id]


    _registry = {
        Key(8, 0, (2,)): Value("A", 0.0, 1, 2),
        Key(8, -1, (1,)): Value("D", 5.0, 2, 1),
        #
        Key(16, 0, (2,)): Value("A", 0.0, 3, 4),
        Key(16, -1, (1,)): Value("D", 5.0, 4, 3),
        #
        Key(7, +1, (1, 1, 2)): Value("A", 5.0, 5, 6),
        Key(7, 0, (1, 1, 1)): Value("D", 0.0, 6, 5),
        #
        Key(7, 0, (1, 2)): Value("A", 0.0, 7, 8),
        Key(7, -1, (1, 1)): Value("D", 5.0, 8, 7),
        #
        Key(7, 0, (3,)): Value("A", 0.0, 9, 10),
        Key(7, -1, (2,)): Value("D", 5.0, 10, 9),
    }

    _resonance_keys_by_id = {
        resonance_type.id: key
        for key, resonance_type in _registry.items()
    }

    @classmethod
    def get_resonance_type(
        cls,
        atomic_number: Literal[8, 16, 7],
        formal_charge: int,
        bond_orders: Tuple[int, ...],
    ):
        bond_orders = tuple(map(int, sorted(bond_orders)))
        key = cls.Key(atomic_number, formal_charge, bond_orders)
        return cls._registry[key]



class FromYamlMixin:

    @classmethod
    def from_yaml_file(cls, *paths, **kwargs):
        import yaml

        yaml_kwargs = {}
        for path in paths:
            with open(str(path), "r") as f:
                dct = yaml.load(f, Loader=yaml.Loader)
                dct = {k.replace("-", "_"): v for k, v in dct.items()}
                yaml_kwargs.update(dct)
        yaml_kwargs.update(kwargs)
        return cls(**yaml_kwargs)


