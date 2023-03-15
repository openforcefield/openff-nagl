"""Bond features for GNN models.

A bond featurization scheme is a tuple of instances of the classes in this
module:

>>> bond_features = (
...     BondIsAromatic(),
...     BondOrder(),
...     BondInRingOfSize(3),
...     BondInRingOfSize(4),
...     BondInRingOfSize(5),
...     BondInRingOfSize(6),
...     ...
... )

The :py:class:`BondFeature` and :py:class:`BondFeatureMeta` classes may be used
to implement your own features.

"""

from typing import ClassVar, Dict, Type

import torch

from openff.nagl.toolkits.openff import get_openff_molecule_bond_indices

from ._base import CategoricalMixin, Feature, FeatureMeta
from ._utils import one_hot_encode

__all__ = [
    "BondFeatureMeta",
    "BondFeature",
    "BondIsAromatic",
    "BondIsInRing",
    "BondInRingOfSize",
    "WibergBondOrder",
    "BondOrder",
]


class BondFeatureMeta(FeatureMeta):
    """Metaclass for registering bond features for string lookup."""

    registry: ClassVar[Dict[str, Type]] = {}


class BondFeature(Feature, metaclass=BondFeatureMeta):
    """Abstract base class for features of bonds."""

    pass


class BondIsAromatic(BondFeature):
    """One-hot encoding for whether the bond is aromatic or not."""

    def _encode(self, molecule) -> torch.Tensor:
        return torch.tensor([bool(bond.is_aromatic) for bond in molecule.bonds])


class BondIsInRing(BondFeature):
    """
    One-hot encoding for whether the bond is in a ring of any size.

    See Also
    --------
    BondInRingOfSize

    """

    def _encode(self, molecule) -> torch.Tensor:
        ring_bonds = {
            tuple(sorted(match))
            for match in molecule.chemical_environment_matches("[*:1]@[*:2]")
        }
        molecule_bonds = get_openff_molecule_bond_indices(molecule)

        tensor = torch.tensor([bool(bond in ring_bonds) for bond in molecule_bonds])
        return tensor


class BondInRingOfSize(BondFeature):
    """
    One-hot encoding for whether the bond is in a ring of the given size.

    The size of the ring is specified by the argument. For a ring of any size,
    see :py:class:`BondIsInRing`. To produce features corresponding to rings of
    multiple sizes, provide this feature multiple times:

    >>> bond_features = (
    ...     BondInRingOfSize(3),
    ...     BondInRingOfSize(4),
    ...     BondInRingOfSize(5),
    ...     BondInRingOfSize(6),
    ...     ...
    ... )

    See Also
    --------
    BondIsInRing, AtomIsInRingOfSize, AtomIsInRing

    """

    ring_size: int

    def _encode(self, molecule) -> torch.Tensor:
        from openff.nagl.toolkits.openff import get_bonds_are_in_ring_size

        is_in_ring = get_bonds_are_in_ring_size(molecule, self.ring_size)
        return torch.tensor(is_in_ring, dtype=int)


class WibergBondOrder(BondFeature):
    """
    The Wiberg fractional bond order of the bond.

    This feature encodes the Wiberg bond order directly, it does not use a
    one-hot encoding.
    """

    def _encode(self, molecule) -> torch.Tensor:
        return torch.tensor([bond.fractional_bond_order for bond in molecule.bonds])


class BondOrder(CategoricalMixin, BondFeature):
    """
    One-hot encoding of the bond order.

    The bond order is also known as the degree of the bond; a single bond has
    bond order 1, a double bond has bond order 2, etc. By default, one-hot
    encodings are provided for all of the formal charges in
    the :py:data:`categories` field. To cover a different list of charges,
    provide that list as an argument to the feature:

    >>> bond_features = (
    ...     BondOrder([1, 2, 3, 4]),
    ...     ...
    ... )
    """

    categories = [1, 2, 3]

    def _encode(self, molecule) -> torch.Tensor:
        return torch.vstack(
            [
                one_hot_encode(int(bond.bond_order), self.categories)
                for bond in molecule.bonds
            ]
        )
