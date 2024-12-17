from typing import Dict, TYPE_CHECKING
import warnings


if TYPE_CHECKING:
    from openff.toolkit.topology.molecule import Molecule
    import torch

FORWARD = "forward"
REVERSE = "reverse"
FEATURE = "feat"


def _get_openff_molecule_information(
    molecule: "Molecule",
) -> Dict[str, "torch.Tensor"]:
    from openff.units import unit
    import torch

    charges = [
        atom.formal_charge.m_as(unit.elementary_charge) for atom in molecule.atoms
    ]
    atomic_numbers = [atom.atomic_number for atom in molecule.atoms]
    return {
        "idx": torch.arange(molecule.n_atoms, dtype=torch.int32),
        "formal_charge": torch.tensor(charges, dtype=torch.int8),
        "atomic_number": torch.tensor(atomic_numbers, dtype=torch.int8),
    }

def _add_xyz_information(
    molecule, molecule_graph
):
    from openff.units import unit
    import torch

    if not molecule.conformers:
        raise ValueError("Molecule does not have coordinates.")
    if len(molecule.conformers) > 1:
        warnings.warn(
            "Molecule has multiple conformers. Using the first one."
        )
    molecule_graph.ndata["xyz"] = torch.tensor(
        molecule.conformers[0].m_as(unit.angstrom),
        dtype=torch.float32,
    )
