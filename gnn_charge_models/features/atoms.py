from typing import TYPE_CHECKING, ClassVar, Dict, List, Type

import numpy as np
import torch

from .base import CategoricalMixin, Feature, FeatureMeta
from .utils import one_hot_encode

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule

__all__ = [
    "AtomFeatureMeta",
    "AtomFeature",
    "AtomicElement",
    "AtomConnectivity",
    "AtomIsAromatic",
    "AtomIsInRing",
    "AtomInRingOfSize",
    "AtomFormalCharge",
    "AtomAverageFormalCharge",
]


class AtomFeatureMeta(FeatureMeta):
    registry: ClassVar[Dict[str, Type]] = {}


class AtomFeature(Feature, metaclass=AtomFeatureMeta):
    pass


class AtomicElement(CategoricalMixin, AtomFeature):
    categories: List[str] = ["H", "C", "N", "O", "F", "Cl", "Br", "S", "P"]

    def _encode(self, molecule: "OFFMolecule") -> torch.Tensor:
        elements = [atom.element for atom in molecule.atoms]
        if not all(isinstance(el, str) for el in elements):
            elements = [el.symbol for el in elements]
        return torch.vstack([one_hot_encode(el, self.categories) for el in elements])


class AtomConnectivity(CategoricalMixin, AtomFeature):
    categories: List[int] = [1, 2, 3, 4]

    def _encode(self, molecule) -> torch.Tensor:
        return torch.vstack(
            [
                one_hot_encode(len(atom.bonds), self.categories)
                for atom in molecule.atoms
            ]
        )


class AtomIsAromatic(AtomFeature):
    def _encode(self, molecule) -> torch.Tensor:
        return torch.tensor([int(atom.is_aromatic) for atom in molecule.atoms])


class AtomIsInRing(AtomFeature):
    def _encode(self, molecule) -> torch.Tensor:
        ring_atoms = [
            index for index, in molecule.chemical_environment_matches("[*r:1]")
        ]
        tensor = torch.zeros(molecule.n_atoms)
        tensor[ring_atoms] = 1
        return tensor


class AtomInRingOfSize(AtomFeature):
    ring_size: int

    def _encode(self, molecule: "OFFMolecule") -> torch.Tensor:
        rdmol = molecule.to_rdkit()

        in_ring_size = [atom.IsInRingSize(self.ring_size) for atom in rdmol.GetAtoms()]
        return torch.tensor(in_ring_size, dtype=int)


class AtomFormalCharge(CategoricalMixin, AtomFeature):
    categories: List[int] = [-3, -2, -1, 0, 1, 2, 3]

    def _encode(self, molecule) -> torch.Tensor:
        from ..utils.openff import get_openff_molecule_formal_charges

        charges = get_openff_molecule_formal_charges(molecule)

        return torch.vstack(
            [one_hot_encode(charge, self.categories) for charge in charges]
        )


class AtomAverageFormalCharge(AtomFeature):
    def _encode(self, molecule: "OFFMolecule") -> torch.Tensor:
        from gnn_charge_models.resonance.resonance import ResonanceEnumerator
        from gnn_charge_models.utils.openff import normalize_molecule

        molecule = normalize_molecule(molecule)
        enumerator = ResonanceEnumerator.from_openff(molecule)
        enumerator.enumerate_resonance_fragments(
            lowest_energy_only=True,
            include_all_transfer_pathways=False,
        )

        formal_charges: List[float] = []
        for rdatoms in enumerator.get_resonance_atoms():
            charges = [atom.GetFormalCharge() for atom in rdatoms]
            charge = np.mean(charges) if charges else 0.0
            formal_charges.append(charge)

        return torch.tensor(formal_charges)
