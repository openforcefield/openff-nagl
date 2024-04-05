"""Atom features for GNN models.

An atom featurization scheme is a tuple of instances of the classes in this
module:

>>> atom_features = (
...     AtomicElement(),
...     AtomHybridization(),
...     AtomConnectivity(),
...     AtomAverageFormalCharge(),
...     AtomGasteigerCharge(),
...     AtomInRingOfSize(3),
...     AtomInRingOfSize(4),
...     AtomInRingOfSize(5),
...     AtomInRingOfSize(6),
...     ...
... )

The :py:class:`AtomFeature` and :py:class:`AtomFeatureMeta` classes may be used
to implement your own features.

"""

import copy
import typing
# from typing import TYPE_CHECKING, ClassVar, Dict, List, Type

import numpy as np
import torch

from openff.nagl.utils._types import HybridizationType
from openff.units import unit
from openff.utilities import requires_package

from ._base import CategoricalMixin, Feature #, FeatureMeta
from ._utils import one_hot_encode

try:
    from pydantic.v1 import validator, Field
except ImportError:
    from pydantic import validator, Field

if typing.TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

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


# class _AtomFeatureMeta(FeatureMeta):
#     """Metaclass for registering atom features for string lookup."""

#     registry: ClassVar[Dict[str, Type]] = {}


class AtomFeature(Feature):#, metaclass=_AtomFeatureMeta):
    """Abstract base class for features of atoms.

    See :py:class:`Feature<openff.nagl.features.Feature>` for details on how to
    implement your own atom features.
    """

    pass


class AtomicElement(CategoricalMixin, AtomFeature):
    """One-hot encodings for specified elements

    By default, one-hot encodings are provided for all of the elements in
    the :py:data:`categories` field. To cover a different list of elements,
    provide that list as an argument to the feature:

    >>> atom_features = (
    ...     AtomicElement(["C", "H", "O", "N", "P", "S"]),
    ...     ...
    ... )
    """

    name: typing.Literal["atomic_element"] = "atomic_element"
    categories: typing.List[str] = ["H", "C", "N", "O", "F", "Cl", "Br", "S", "P", "I"]
    """Elements to provide one-hot encodings for."""

    def _encode(self, molecule: "Molecule") -> torch.Tensor:
        try:
            elements = [atom.element for atom in molecule.atoms]
        except AttributeError:
            elements = [atom.symbol for atom in molecule.atoms]
        if not all(isinstance(el, str) for el in elements):
            elements = [el.symbol for el in elements]
        return torch.vstack([one_hot_encode(el, self.categories) for el in elements])


class AtomHybridization(CategoricalMixin, AtomFeature):
    """
    One-hot encodings for the specified atomic orbital hybridization modes.
    """
    name: typing.Literal["atom_hybridization"] = "atom_hybridization"

    categories: typing.List[HybridizationType] = [
        HybridizationType.OTHER,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
    ]
    """The supported hybridization modes."""

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
    """
    One-hot encodings for the number of other atoms this atom is connected to.

    By default, one-hot encodings are provided for all of the connectivities in
    the :py:data:`categories` field. To cover a different list of numbers,
    provide that list as an argument to the feature:

    >>> atom_features = (
    ...     AtomConnectivity([1, 2, 3, 4, 5, 6]),
    ...     ...
    ... )
    """
    name: typing.Literal["atom_connectivity"] = "atom_connectivity"

    categories: typing.List[int] = [1, 2, 3, 4]
    """Connectivities to provide one-hot encodings for."""

    def _encode(self, molecule) -> torch.Tensor:
        return torch.vstack(
            [
                one_hot_encode(len(atom.bonds), self.categories)
                for atom in molecule.atoms
            ]
        )


class AtomIsAromatic(AtomFeature):
    """One-hot encoding for whether the atom is aromatic or not."""

    name: typing.Literal["atom_is_aromatic"] = "atom_is_aromatic"

    def _encode(self, molecule) -> torch.Tensor:
        return torch.tensor([int(atom.is_aromatic) for atom in molecule.atoms])


class AtomIsInRing(AtomFeature):
    """
    One-hot encoding for whether the atom is in a ring of any size.

    See Also
    --------
    AtomInRingOfSize, BondIsInRingOfSize, BondIsInRing

    """
    name: typing.Literal["atom_is_in_ring"] = "atom_is_in_ring"

    def _encode(self, molecule) -> torch.Tensor:
        ring_atoms = [
            index for index, in molecule.chemical_environment_matches("[*r:1]")
        ]
        tensor = torch.zeros(molecule.n_atoms, dtype=bool)
        tensor[ring_atoms] = True
        return tensor


class AtomInRingOfSize(AtomFeature):
    """
    One-hot encoding for whether the atom is in a ring of the given size.

    The size of the ring is specified by the argument. For a ring of any size,
    see :py:class:`AtomIsInRing`. To produce features corresponding to rings of
    multiple sizes, provide this feature multiple times:

    >>> atom_features = (
    ...     AtomInRingOfSize(3),
    ...     AtomInRingOfSize(4),
    ...     AtomInRingOfSize(5),
    ...     AtomInRingOfSize(6),
    ...     ...
    ... )

    See Also
    --------
    AtomIsInRing, BondIsInRingOfSize, BondIsInRing

    """
    name: typing.Literal["atom_in_ring_of_size"] = "atom_in_ring_of_size"

    ring_size: int
    """The size of the ring that this feature describes."""

    def _encode(self, molecule: "Molecule") -> torch.Tensor:
        from openff.nagl.toolkits.openff import get_atoms_are_in_ring_size

        in_ring_size = get_atoms_are_in_ring_size(molecule, self.ring_size)
        # rdmol = openff_to_rdkit(molecule)

        # in_ring_size = [atom.IsInRingSize(self.ring_size) for atom in rdmol.GetAtoms()]
        return torch.tensor(in_ring_size, dtype=int)


class AtomFormalCharge(CategoricalMixin, AtomFeature):
    """
    One-hot encoding of the formal charge on an atom.

    By default, one-hot encodings are provided for all of the formal charges in
    the :py:data:`categories` field. To cover a different list of charges,
    provide that list as an argument to the feature:

    >>> atom_features = (
    ...     AtomFormalCharge([-4, -3, -2, -1, 0, 1, 2, 3, 4]),
    ...     ...
    ... )
    """

    name: typing.Literal["atom_formal_charge"] = "atom_formal_charge"

    categories: typing.List[int] = [-3, -2, -1, 0, 1, 2, 3]

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
    """
    The formal charge of the atom, averaged over all resonance forms.

    This feature encodes the average formal charge directly, it does not use a
    one-hot encoding.
    """
    name: typing.Literal["atom_average_formal_charge"] = "atom_average_formal_charge"

    def _encode(self, molecule: "Molecule") -> torch.Tensor:
        from openff.nagl.utils.resonance import enumerate_resonance_forms
        from openff.nagl.toolkits.openff import normalize_molecule

        molecule = normalize_molecule(molecule)
        resonance_forms = enumerate_resonance_forms(
            molecule,
            lowest_energy_only=True,
            include_all_transfer_pathways=False,
            as_dicts=True,
            as_fragments=True,
        )
        formal_charges: typing.List[float] = []
        for index in range(molecule.n_atoms):
            charges = [
                graph["atoms"][index]["formal_charge"]
                for graph in resonance_forms
                if index in graph["atoms"]
            ]
            if not charges:
                charges = [molecule.atoms[index].formal_charge]

            charges = [q.m_as(unit.elementary_charge) for q in charges]
            charge = np.mean(charges)
            formal_charges.append(charge)

        return torch.tensor(formal_charges)


class AtomGasteigerCharge(AtomFeature):
    """
    The Gasteiger partial charge of the atom.

    This feature encodes the Gasteiger charge directly, it does not use a
    one-hot encoding.
    """
    name: typing.Literal["atom_gasteiger_charge"] = "atom_gasteiger_charge"

    def _encode(self, molecule) -> torch.Tensor:
        from openff.units import unit

        molecule = copy.deepcopy(molecule)
        molecule.assign_partial_charges("gasteiger")
        charges = molecule.partial_charges.m_as(unit.elementary_charge)
        return torch.tensor(charges)

class AtomElementPeriod(CategoricalMixin, AtomFeature):
    name: typing.Literal["atom_element_period"] = "atom_element_period"

    categories: typing.List[int] = [1, 2, 3, 4, 5]

    def _encode(self, molecule) -> torch.Tensor:
        PERIODS = {
            "H": 1,
            "He": 1,
            "C": 2,
            "N": 2,
            "O": 2,
            "F": 2,
            "Si": 3,
            "P": 3,
            "S": 3,
            "Cl": 3,
            "Br": 4,
            "I": 5
        }

        try:
            elements = [atom.element for atom in molecule.atoms]
        except AttributeError:
            elements = [atom.symbol for atom in molecule.atoms]
        if not all(isinstance(el, str) for el in elements):
            elements = [el.symbol for el in elements]

        periods = [PERIODS[x] for x in elements]
        return torch.vstack(
            [
                one_hot_encode(p, self.categories)
                for p in periods
            ]
        )


class AtomElementGroup(CategoricalMixin, AtomFeature):
    name: typing.Literal["atom_element_group"] = "atom_element_group"

    categories: typing.List[int] = [1, 14, 15, 16, 17]

    def _encode(self, molecule) -> torch.Tensor:
        GROUPS = {
            "H": 1,
            "C": 14,
            "N": 15,
            "O": 16,
            "F": 17,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
            "Br": 17,
            "I":17
        }

        try:
            elements = [atom.element for atom in molecule.atoms]
        except AttributeError:
            elements = [atom.symbol for atom in molecule.atoms]
        if not all(isinstance(el, str) for el in elements):
            elements = [el.symbol for el in elements]

        groups = [GROUPS[x] for x in elements]
        return torch.vstack(
            [
                one_hot_encode(p, self.categories)
                for p in groups
            ]
        )


class AtomTotalBondOrder(AtomFeature):
    name: typing.Literal["atom_total_bond_order"] = "atom_total_bond_order"

    def _encode(self, molecule) -> torch.Tensor:

        bond_orders = [
            sum(bond.bond_order for bond in atom.bonds)
            for atom in molecule.atoms
        ]

        return torch.tensor(bond_orders)




class AtomElectronegativityAllredRochow(AtomFeature):
    name: typing.Literal["atom_electronegativity_allred_rochow"] = "atom_electronegativity_allred_rochow"

    def _encode(self, molecule) -> torch.Tensor:
        from ._data import ALLRED_ROCHOW_ELECTRONEGATIVITY

        electronegativities = [
            ALLRED_ROCHOW_ELECTRONEGATIVITY[atom.atomic_number]
            for atom in molecule.atoms
        ]

        return torch.tensor(electronegativities)


class AtomElectronAffinity(AtomFeature):
    name: typing.Literal["atom_electron_affinity"] = "atom_electron_affinity"

    def _encode(self, molecule) -> torch.Tensor:
        from ._data import ELECTRON_AFFINITY

        affinities = [
            ELECTRON_AFFINITY[atom.atomic_number] for atom in molecule.atoms
        ]

        return torch.tensor(affinities)

class AtomElectrophilicity(AtomFeature):
    name: typing.Literal["atom_electrophilicity"] = "atom_electrophilicity"

    def _encode(self, molecule) -> torch.Tensor:
        from ._data import ELECTROPHILICITIES

        affinities = [
            ELECTROPHILICITIES[atom.atomic_number] for atom in molecule.atoms
        ]

        return torch.tensor(affinities)


AtomFeatureType = typing.Union[
    AtomicElement,
    AtomHybridization,
    AtomConnectivity,
    AtomIsAromatic,
    AtomIsInRing,
    AtomInRingOfSize,
    AtomFormalCharge,
    AtomAverageFormalCharge,
    AtomGasteigerCharge,
    AtomElementPeriod,
    AtomElementGroup,
    AtomTotalBondOrder,
    AtomElectronAffinity,
    AtomElectrophilicity,
    AtomElectronegativityAllredRochow
]

DiscriminatedAtomFeatureType = typing.Annotated[
    AtomFeatureType, Field(..., discriminator="name")
]