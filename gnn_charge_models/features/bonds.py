from typing import ClassVar, List, Tuple, Dict, Type

from pydantic.main import ModelMetaclass
import torch

from .feature import Feature, CategoricalMixin, FeatureMeta
from .utils import one_hot_encode
from ..utils import get_openff_molecule_bond_indices

class BondFeatureMeta(FeatureMeta):
    registry: ClassVar[Dict[str, Type]] = {}


class BondFeature(Feature, metaclass=BondFeatureMeta):
    pass


class BondIsAromatic(BondFeature):

    def _encode(self, molecule) -> torch.Tensor:
        return torch.tensor([
            int(bond.is_aromatic)
            for bond in molecule.bonds
        ])


class BondIsInRing(BondFeature):

    def _encode(self, molecule) -> torch.Tensor:
        ring_bonds = {
            tuple(sorted(match))
            for match in molecule.chemical_environment_matches("[*:1]@[*:2]")
        }
        molecule_bonds = get_openff_molecule_bond_indices(molecule)

        tensor = torch.tensor([
            int(bond in ring_bonds)
            for bond in molecule_bonds
        ])
        return tensor


class WibergBondOrder(BondFeature):
    def _encode(self, molecule) -> torch.Tensor:
        return torch.tensor([
            bond.fractional_bond_order
            for bond in molecule.bonds
        ])


class BondOrder(CategoricalMixin, BondFeature):
    categories = [1, 2, 3]

    def _encode(self, molecule) -> torch.Tensor:
        return torch.vstack([
            one_hot_encode(int(bond.bond_order), self.categories)
            for bond in molecule.bonds
        ])
