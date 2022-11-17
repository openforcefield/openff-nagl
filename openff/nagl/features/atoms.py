from typing import TYPE_CHECKING, ClassVar, Dict, List, Type

import numpy as np
import torch
from pydantic import validator

from openff.nagl.utils.types import HybridizationType
from openff.utilities import requires_package

from .base import CategoricalMixin, Feature, FeatureMeta
from .utils import one_hot_encode


if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule

__all__ = [
    "AtomFeatureMeta",
    "AtomFeature",
    "AtomicElement",
    "AtomConnectivity",
    "AtomHybridization",
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
        from openff.nagl.utils.openff import get_molecule_hybridizations

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
        from openff.nagl.utils.openff import openff_to_rdkit
        rdmol = openff_to_rdkit(molecule)

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
        from openff.nagl.resonance.resonance import ResonanceEnumerator
        from openff.nagl.utils.openff import normalize_molecule

        molecule = normalize_molecule(molecule, check_output=False)
        enumerator = ResonanceEnumerator(molecule)
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


class AtomGasteigerCharge(AtomFeature):
    def _encode(self, molecule) -> torch.Tensor:
        from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

        rdmol = molecule.to_rdkit()
        ComputeGasteigerCharges(rdmol)

        charges = [
            rdatom.GetProp("_GasteigerCharge")
            for rdatom in rdmol.GetAtoms()
        ]

        return torch.tensor(charges)


class AtomMorganFingerprint(AtomFeature):

    radius: int = 2
    # n_bits: int = 1024

    _feature_length: int = 1024

    @requires_package("rdkit")
    def _encode(self, molecule: "OFFMolecule") -> torch.Tensor:
        from rdkit.Chem import rdMolDescriptors

        rdmol = molecule.to_rdkit()
        fingerprints = []
        for atom in rdmol.GetAtoms():
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                rdmol,
                radius=self.radius,
                nBits=self._feature_length,
                fromAtoms=[atom.GetIdx()]
            )
            fp.ToList()
            fingerprints.append(fp)
        
        feature = torch.tensor(fingerprints)
        return feature