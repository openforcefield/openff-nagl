from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import torch
    from openff.toolkit.topology.molecule import Molecule

FORWARD = "forward"
REVERSE = "reverse"
FEATURE = "feat"


def _get_openff_molecule_information(
    molecule: "Molecule",
) -> Dict[str, "torch.Tensor"]:
    import torch
    from openff.units import unit

    charges = [
        atom.formal_charge.m_as(unit.elementary_charge) for atom in molecule.atoms
    ]
    atomic_numbers = [atom.atomic_number for atom in molecule.atoms]
    return {
        "idx": torch.arange(molecule.n_atoms, dtype=torch.int32),
        "formal_charge": torch.tensor(charges, dtype=torch.int8),
        "atomic_number": torch.tensor(atomic_numbers, dtype=torch.int8),
    }
