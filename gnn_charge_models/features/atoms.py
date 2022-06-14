from typing import ClassVar, List, Tuple, Dict, Type

from pydantic.main import ModelMetaclass
import torch

from .feature import Feature, CategoricalMixin, FeatureMeta
from .utils import one_hot_encode


class AtomFeatureMeta(FeatureMeta):
    registry: ClassVar[Dict[str, Type]] = {}


class AtomFeature(Feature, metaclass=AtomFeatureMeta):
    pass


class AtomicElement(CategoricalMixin, AtomFeature):
    feature_name: ClassVar[str] = "atomic_element"
    categories: List[str] = ["H", "C", "N", "O", "F", "Cl", "Br", "S", "P"]

    def _encode(self, molecule) -> torch.Tensor:
        return torch.vstack([
            one_hot_encode(atom.element, self.categories)
            for atom in molecule.atoms
        ])


class AtomConnectivity(CategoricalMixin, AtomFeature):
    feature_name: ClassVar[str] = "atom_connectivity"
    categories: List[int] = [1, 2, 3, 4]

    def _encode(self, molecule) -> torch.Tensor:
        return torch.vstack([
            one_hot_encode(len(atom.bonds), self.categories)
            for atom in molecule.atoms
        ])


class AtomIsAromatic(AtomFeature):
    feature_name: ClassVar[str] = "atom_is_aromatic"

    def _encode(self, molecule) -> torch.Tensor:
        return torch.tensor([
            int(atom.is_aromatic)
            for atom in molecule.atoms
        ])


class AtomIsInRing(AtomFeature):
    feature_name: ClassVar[str] = "atom_is_in_ring"

    def _encode(self, molecule) -> torch.Tensor:
        ring_atoms = [
            index for index, in molecule.chemical_environment_matches("[*r:1]")
        ]
        tensor = torch.zeros(molecule.n_atoms)
        tensor[ring_atoms] = 1
        return tensor


class AtomFormalCharge(CategoricalMixin, AtomFeature):
    feature_name: ClassVar[str] = "atom_formal_charge"
    categories: List[int] = [-3, -2, -1, 0, 1, 2, 3]

    def _encode(self, molecule) -> torch.Tensor:
        from ..utils.openff import get_openff_molecule_formal_charges
        charges = get_openff_molecule_formal_charges(molecule)

        return torch.vstack([
            one_hot_encode(charge, self.categories)
            for charge in charges
        ])
