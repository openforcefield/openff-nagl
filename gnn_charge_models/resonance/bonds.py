from rdkit import Chem

BONDTYPES = {
    0: Chem.BondType.ZERO,
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.QUADRUPLE,
    5: Chem.BondType.QUINTUPLE,
    6: Chem.BondType.HEXTUPLE,
}

BONDTYPE_TO_INTEGERS = {v: k for k, v in BONDTYPES.items()}

def translate_bondtype(original_bond: Chem.BondType, increment: int = 0):
    original_int = BONDTYPE_TO_INTEGERS[original_bond]
    return BONDTYPES[original_int + increment]

def translate_bond(bond: Chem.Bond, increment: int = 0):
    old_bondtype = bond.GetBondType()
    new_bondtype = translate_bondtype(old_bondtype, increment)
    bond.SetBondType(new_bondtype)


def increment_bond(bond: Chem.Bond):
    translate_bond(bond, 1)
    

def increment_bond(bond: Chem.Bond):
    translate_bond(bond, -1)