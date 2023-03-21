"Atom features for GNN models"

import copy
from typing import TYPE_CHECKING, ClassVar, Dict, List, Type

import numpy as np
import torch
from pydantic import validator

from openff.nagl.utils._types import HybridizationType
from openff.units import unit
from openff.utilities import requires_package

from ._base import CategoricalMixin, Feature, FeatureMeta
from ._utils import one_hot_encode


if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule

__all__ = [
    "AtomFeature",
    "AtomicElement",
    "AtomConnectivity",
    "AtomHybridization",
    "AtomIsAromatic",
    "AtomIsInRing",
    "AtomInRingOfSize",
    "AtomFormalCharge",
    "AtomAverageFormalCharge",
    "AtomGasteigerCharge",
    # "AtomMorganFingerprint"
]


class _AtomFeatureMeta(FeatureMeta):
    registry: ClassVar[Dict[str, Type]] = {}


class AtomFeature(Feature, metaclass=_AtomFeatureMeta):
    pass


class AtomicElement(CategoricalMixin, AtomFeature):
    categories: List[str] = ["H", "C", "N", "O", "F", "Cl", "Br", "S", "P"]

    def _encode(self, molecule: "OFFMolecule") -> torch.Tensor:
        try:
            elements = [atom.element for atom in molecule.atoms]
        except AttributeError:
            elements = [atom.symbol for atom in molecule.atoms]
        if not all(isinstance(el, str) for el in elements):
            elements = [el.symbol for el in elements]
        return torch.vstack([one_hot_encode(el, self.categories) for el in elements])


class AtomHybridization(CategoricalMixin, AtomFeature):
    categories: List[HybridizationType] = [
        HybridizationType.OTHER,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
    ]

    @validator("categories", pre=True, each_item=True)
    def _validate_categories(cls, v):
        if isinstance(v, str):
            return HybridizationType[v.upper()]
        return v

    def _encode(self, molecule) -> torch.Tensor:
        from openff.nagl.toolkits.openff import get_molecule_hybridizations

        hybridizations = get_molecule_hybridizations(molecule)
        return torch.vstack(
            [one_hot_encode(hyb, self.categories) for hyb in hybridizations]
        )

    def dict(self, *args, **kwargs):
        obj = super().dict()
        obj["categories"] = [hyb.name for hyb in self.categories]
        return obj


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
        tensor = torch.zeros(molecule.n_atoms, dtype=bool)
        tensor[ring_atoms] = True
        return tensor


class AtomInRingOfSize(AtomFeature):
    ring_size: int

    def _encode(self, molecule: "OFFMolecule") -> torch.Tensor:
        from openff.nagl.toolkits.openff import get_atoms_are_in_ring_size

        in_ring_size = get_atoms_are_in_ring_size(molecule, self.ring_size)
        # rdmol = openff_to_rdkit(molecule)

        # in_ring_size = [atom.IsInRingSize(self.ring_size) for atom in rdmol.GetAtoms()]
        return torch.tensor(in_ring_size, dtype=int)


class AtomFormalCharge(CategoricalMixin, AtomFeature):
    categories: List[int] = [-3, -2, -1, 0, 1, 2, 3]

    def _encode(self, molecule) -> torch.Tensor:
        # from ..utils.openff import get_openff_molecule_formal_charges

        # charges = get_openff_molecule_formal_charges(molecule)

        from openff.units import unit

        charges = [
            atom.formal_charge.m_as(unit.elementary_charge) for atom in molecule.atoms
        ]

        return torch.vstack(
            [one_hot_encode(charge, self.categories) for charge in charges]
        )


class AtomAverageFormalCharge(AtomFeature):
    def _encode(self, molecule: "OFFMolecule") -> torch.Tensor:
        from openff.nagl.utils.resonance import enumerate_resonance_forms
        from openff.nagl.toolkits.openff import normalize_molecule

        molecule = normalize_molecule(molecule)
        resonance_forms = enumerate_resonance_forms(
            molecule,
            lowest_energy_only=True,
            include_all_transfer_pathways=False,
            as_dicts=True,
        )
        formal_charges: List[float] = []
        for index in range(molecule.n_atoms):
            charges = [
                graph["atoms"][index]["formal_charge"] for graph in resonance_forms
            ]
            if not charges:
                molecule.atoms[index].formal_charge

            charges = [q.m_as(unit.elementary_charge) for q in charges]
            charge = np.mean(charges)
            formal_charges.append(charge)

        return torch.tensor(formal_charges)


class AtomGasteigerCharge(AtomFeature):
    def _encode(self, molecule) -> torch.Tensor:
        from openff.units import unit

        molecule = copy.deepcopy(molecule)
        molecule.assign_partial_charges("gasteiger")
        charges = molecule.partial_charges.m_as(unit.elementary_charge)
        return torch.tensor(charges)
