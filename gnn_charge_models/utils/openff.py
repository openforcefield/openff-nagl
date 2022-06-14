from typing import List, Union, TYPE_CHECKING, Tuple, Dict

import numpy as np
import torch

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule


def get_openff_molecule_bond_indices(molecule: "OFFMolecule") -> List[Tuple[int, int]]:
    return [
        tuple(sorted((bond.atom1_index, bond.atom2_index)))
        for bond in molecule.bonds
    ]


def get_openff_molecule_formal_charges(molecule: "OFFMolecule") -> List[float]:
    from openff.toolkit.topology.molecule import unit as off_unit
    return [
        atom.formal_charge.value_in_unit(off_unit.elementary_charge)
        for atom in molecule.atoms
    ]


def get_openff_molecule_information(molecule: "OFFMolecule") -> Dict[str, "torch.Tensor"]:
    charges = get_openff_molecule_formal_charges(molecule)
    atomic_numbers = [atom.atomic_number for atom in molecule.atoms]
    return {
        "idx": torch.arange(molecule.n_atoms, dtype=torch.int32),
        "formal_charge": torch.tensor(charges, dtype=torch.int8),
        "atomic_number": torch.tensor(atomic_numbers, dtype=torch.int8),
    }
