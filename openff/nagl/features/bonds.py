"""Bond features for GNN models"""

from typing import ClassVar, Dict, Type

import torch

from openff.nagl.toolkits.openff import get_openff_molecule_bond_indices

from ._base import CategoricalMixin, Feature, FeatureMeta
from ._utils import one_hot_encode

__all__ = [
    "BondFeature",
    "BondIsAromatic",
    "BondIsInRing",
    "BondInRingOfSize",
    "WibergBondOrder",
    "BondOrder",
]


class _BondFeatureMeta(FeatureMeta):
    registry: ClassVar[Dict[str, Type]] = {}


class BondFeature(Feature, metaclass=_BondFeatureMeta):
    pass


class BondIsAromatic(BondFeature):
    def _encode(self, molecule) -> torch.Tensor:
        return torch.tensor([bool(bond.is_aromatic) for bond in molecule.bonds])


class BondIsInRing(BondFeature):
    def _encode(self, molecule) -> torch.Tensor:
        ring_bonds = {
            tuple(sorted(match))
            for match in molecule.chemical_environment_matches("[*:1]@[*:2]")
        }
        molecule_bonds = get_openff_molecule_bond_indices(molecule)

        tensor = torch.tensor([bool(bond in ring_bonds) for bond in molecule_bonds])
        return tensor


class BondInRingOfSize(BondFeature):
    ring_size: int

    def _encode(self, molecule) -> torch.Tensor:
        from openff.nagl.toolkits.openff import get_bonds_are_in_ring_size

        is_in_ring = get_bonds_are_in_ring_size(molecule, self.ring_size)
        return torch.tensor(is_in_ring, dtype=int)


class WibergBondOrder(BondFeature):
    def _encode(self, molecule) -> torch.Tensor:
        return torch.tensor([bond.fractional_bond_order for bond in molecule.bonds])


class BondOrder(CategoricalMixin, BondFeature):
    categories = [1, 2, 3]

    def _encode(self, molecule) -> torch.Tensor:
        return torch.vstack(
            [
                one_hot_encode(int(bond.bond_order), self.categories)
                for bond in molecule.bonds
            ]
        )
