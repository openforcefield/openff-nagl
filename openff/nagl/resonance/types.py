import enum
from typing import Dict, Literal, NamedTuple, Tuple


class ResonanceAtomType(enum.Enum):
    Acceptor = "A"
    Donor = "D"


_ATOMIC_NUMBERS = {
    "O": 8,
    "S": 16,
    "N": 7,
}


class ResonanceTypeKey(NamedTuple):
    """A convenient data structure for storing information used to recognize a possible
    resonance atom type by."""

    element: Literal["O", "S", "N"]

    formal_charge: int
    bond_orders: Tuple[int, ...]

    @property
    def atomic_number(self):
        return _ATOMIC_NUMBERS[self.element]


class ResonanceTypeValue(NamedTuple):
    """A convenient data structure for storing information about a possible resonance
    atom type in."""

    type: ResonanceAtomType
    energy: float
    id: int
    conjugate_id: int

    def get_conjugate_key(self):
        return RESONANCE_KEYS_BY_ID[self.conjugate_id]


RESONANCE_TYPES: Dict[ResonanceTypeKey, ResonanceTypeValue] = {
    ResonanceTypeKey("O", 0, (2,)): ResonanceTypeValue("A", 0.0, 1, 2),
    ResonanceTypeKey("O", -1, (1,)): ResonanceTypeValue("D", 5.0, 2, 1),
    #
    ResonanceTypeKey("S", 0, (2,)): ResonanceTypeValue("A", 0.0, 3, 4),
    ResonanceTypeKey("S", -1, (1,)): ResonanceTypeValue("D", 5.0, 4, 3),
    #
    ResonanceTypeKey("N", +1, (1, 1, 2)): ResonanceTypeValue("A", 5.0, 5, 6),
    ResonanceTypeKey("N", 0, (1, 1, 1)): ResonanceTypeValue("D", 0.0, 6, 5),
    #
    ResonanceTypeKey("N", 0, (1, 2)): ResonanceTypeValue("A", 0.0, 7, 8),
    ResonanceTypeKey("N", -1, (1, 1)): ResonanceTypeValue("D", 5.0, 8, 7),
    #
    ResonanceTypeKey("N", 0, (3,)): ResonanceTypeValue("A", 0.0, 9, 10),
    ResonanceTypeKey("N", -1, (2,)): ResonanceTypeValue("D", 5.0, 10, 9),
}

RESONANCE_KEYS_BY_ID: Dict[int, ResonanceTypeValue] = {
    value.id: key for key, value in RESONANCE_TYPES.items()
}


def get_resonance_type(
    element: Literal["O", "S", "N"],
    formal_charge: int,
    bond_orders: Tuple[int, ...],
) -> ResonanceTypeValue:
    """Get the resonance type for a given element, formal charge, and bond orders."""
    return RESONANCE_TYPES[ResonanceTypeKey(element, formal_charge, bond_orders)]
