"""This module defines the data models used to store the data in the database."""

import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple, ClassVar

import numpy as np
from openff.units import unit as openff_unit
from pydantic import Field, validator

from openff.nagl.base.array import Array
from openff.nagl.base.base import ImmutableModel
from openff.nagl.utils.openff import map_indexed_smiles

if TYPE_CHECKING:
    import openff.toolkit


class Record(ImmutableModel):
    class Config(ImmutableModel.Config):
        orm_mode = True


class ChargeMethod(str):
    _openff_conversion = {"am1": "am1-mulliken", "am1bcc": "am1bcc"}

    def to_openff_method(self) -> str:
        key = self.lower()
        return self._openff_conversion.get(key, key)

class WibergBondOrderMethod(str):
    _openff_conversion = {"am1": "am1-wiberg"}

    def to_openff_method(self) -> str:
        key = self.lower()
        return self._openff_conversion.get(key, key)


class PartialChargeRecord(Record):
    """A record of the partial charges calculated for a conformer using a specific method"""

    method: ChargeMethod
    values: Array[float]

    @validator("values", pre=True)
    def _validate_type(cls, v):
        if hasattr(v, "value_in_unit"):
            from openff.toolkit.topology.molecule import unit as toolkit_unit

            v = v.value_in_unit(toolkit_unit.elementary_charge)
        elif isinstance(v, openff_unit.Quantity):
            v = v.m_as(openff_unit.elementary_charge)
        return v

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.values[key]
        return type(self)(method=self.method, values=self.values[key])


class WibergBondOrder(NamedTuple):
    """A single Wiberg bond order for a bond"""

    atom_index_1: int
    atom_index_2: int
    bond_order: float

    @classmethod
    def from_openff(cls, openff_bond):
        return cls(
            atom_index_1=openff_bond.atom1_index,
            atom_index_2=openff_bond.atom2_index,
            bond_order=openff_bond.fractional_bond_order,
        )

    def map_to(self, mapping: Dict[int, int]):
        return type(self)(
            atom_index_1=mapping[self.atom_index_1],
            atom_index_2=mapping[self.atom_index_2],
            bond_order=self.bond_order,
        )

    @property
    def sorted_atom_indices(self):
        return tuple(sorted((self.atom_index_1, self.atom_index_2)))


class WibergBondOrderRecord(Record):
    """A record of the Wiberg bond orders calculated for a conformer using a specific method"""

    method: WibergBondOrderMethod = "am1"
    values: List[WibergBondOrder]

    def map_to(self, mapping: Dict[int, int]):
        return type(self)(
            method=self.method,
            values=[bond.map_to(mapping) for bond in self.values],
        )


class ConformerRecord(Record):
    """A record which stores the coordinates of a molecule in a particular conformer,
    as well as sets of partial charges and WBOs computed using this conformer and
    for different methods."""

    coordinates: Array[float] = Field(
        ...,
        description=(
            "The coordinates [Angstrom] of this conformer " "with shape=(n_atoms, 3)."
        ),
    )
    partial_charges: Dict[ChargeMethod, PartialChargeRecord] = {}
    bond_orders: Dict[WibergBondOrderMethod, WibergBondOrderRecord] = {}

    @validator("coordinates")
    def _validate_coordinate_type(cls, v):
        if hasattr(v, "value_in_unit"):
            from openff.toolkit.topology.molecule import unit as toolkit_unit

            v = v.value_in_unit(toolkit_unit.angstrom)
        elif isinstance(v, openff_unit.Quantity):
            v = v.m_as(openff_unit.angstrom)
        return v

    @validator("coordinates")
    def _validate_coordinates(cls, v):
        try:
            v = v.reshape((-1, 3))
        except ValueError:
            raise ValueError(
                "coordinates must be re-shapable to `(n_atoms, 3)`")
        v.flags.writeable = False
        return v

    @validator("partial_charges", "bond_orders", pre=True)
    def _validate_unique_methods(cls, v, field):
        if not isinstance(v, dict):
            new_v = {}
            for v_ in v:
                new_v[v_.method] = v_
            v = new_v

        value_classes = tuple([x.type_ for x in field.sub_fields])
        new_v = {}
        for k_, v_ in v.items():
            if not isinstance(v_, (dict, *value_classes)):
                v_ = {"method": k_, "values": v_}
            new_v[k_] = v_
        v = new_v
        return v

    def map_to(self, mapping: Dict[int, int]) -> "ConformerRecord":
        """Map the conformer to a new set of atom indices"""
        index_array = np.array(list(mapping.values()))
        inverse_map = {v: k for k, v in mapping.items()}
        return type(self)(
            coordinates=self.coordinates[index_array],
            partial_charges={
                method: charges[index_array]
                for method, charges in self.partial_charges.items()
            },
            bond_orders={
                method: bond_orders.map_to(inverse_map)
                for method, bond_orders in self.bond_orders.items()
            },
        )


class MoleculeRecord(Record):
    """A record which contains information for a labelled molecule. This may include the
    coordinates of the molecule in different conformers, and partial charges / WBOs
    computed for those conformers."""

    mapped_smiles: str = Field(
        ...,
        description="The mapped SMILES string for the molecule with hydrogens specified",
    )
    conformers: List[ConformerRecord] = Field(
        ...,
        description=(
            "Conformers associated with the molecule. "
            "Each conformer contains computed, labelled properties, "
            "such as partial charges and Wiberg bond orders"
        ),
    )

    _partial_charge_method_name: ClassVar[str] = "partial_charge_method"
    _bond_order_method_name: ClassVar[str] = "bond_order_method"

    @property
    def smiles(self):
        return self.mapped_smiles

    @classmethod
    def from_precomputed_openff(
        cls,
        molecule: "openff.toolkit.topology.Molecule",
        partial_charge_method: str = None,
        bond_order_method: str = None,
    ):
        from openff.nagl.nn.label import LabelPrecomputedMolecule
        from openff.nagl.utils.openff import get_coordinates_in_angstrom

        if not len(molecule.conformers) == 1:
            raise ValueError(
                "The molecule must have exactly one conformer to be "
                "converted to a MoleculeRecord "
                "using Molecule.from_precomputed_openff. "
                "Try using Molecule.from_openff and generating conformers "
                "instead"
            )

        labeller = LabelPrecomputedMolecule(
            partial_charge_method=partial_charge_method,
            bond_order_method=bond_order_method,
        )
        labels = labeller(molecule)

        charges = {}
        if labeller.partial_charge_label in labels:
            charges[partial_charge_method] = labels[labeller.partial_charge_label].numpy()
        bonds = {}
        if labeller.bond_order_label in labels:
            bonds[bond_order_method] = labels[labeller.bond_order_label].numpy()

        conformer_record = ConformerRecord(
            coordinates=get_coordinates_in_angstrom(molecule.conformers[0]),
            partial_charges=charges,
            bond_orders=bonds,
        )
        record = cls(
            mapped_smiles=molecule.to_smiles(mapped=True, isomeric=True),
            conformers=[conformer_record],
        )
        return record

    @classmethod
    def from_openff(
        cls,
        openff_molecule: "openff.toolkit.topology.Molecule",
        partial_charge_methods: Tuple[ChargeMethod] = tuple(),
        bond_order_methods: Tuple[WibergBondOrderMethod] = tuple(),
        generate_conformers: bool = False,
        n_conformer_pool: int = 500,
        n_conformers: int = 10,
        rms_cutoff: float = 0.05,
    ):
        """Create a MoleculeRecord from an OpenFF molecule

        Parameters
        ----------
        openff_molecule
            The molecule to create a record for
        partial_charge_methods
            The methods used to compute partial charges
        bond_order_methods
            The methods used to compute Wiberg bond orders
        generate_conformers
            Whether to generate new conformers to overwrite existing ones
        n_conformer_pool
            The number of conformers to generate as a first step.
            ELF conformers will be selected from these.
        n_conformers
            The number of conformers to select from the pool
        rms_cutoff
            The minimum RMS cutoff difference between conformers
        """
        from openff.toolkit.topology.molecule import unit as off_unit

        from openff.nagl.storage.record import (
            ConformerRecord,
            PartialChargeRecord,
            WibergBondOrder,
            WibergBondOrderRecord,
        )
        from openff.nagl.utils.openff import (
            get_coordinates_in_angstrom,
            get_unitless_charge,
        )

        partial_charge_methods = [
            ChargeMethod(method) for method in partial_charge_methods
        ]
        bond_order_methods = [
            WibergBondOrderMethod(method) for method in bond_order_methods
        ]

        molecule = copy.deepcopy(openff_molecule)

        if generate_conformers:
            molecule.generate_conformers(
                n_conformers=n_conformer_pool,
                rms_cutoff=rms_cutoff * off_unit.angstrom,
            )
            molecule.apply_elf_conformer_selection(limit=n_conformers)
        
        elif not molecule.conformers:
            raise ValueError(
                "Molecule must have conformers to create a record. "
                "Either pass in a molecule with conformers "
                "or set generate_conformers=True"
            )

        conformer_records = []
        for conformer in molecule.conformers:
            charge_sets = {}
            for method in partial_charge_methods:
                molecule.assign_partial_charges(
                    method.to_openff_method(),
                    use_conformers=[conformer],
                )
                charge_sets[method] = PartialChargeRecord(
                    method=method,
                    values=[get_unitless_charge(x)
                            for x in molecule.partial_charges],
                )

            bond_order_sets = {}
            for method in bond_order_methods:
                molecule.assign_fractional_bond_orders(
                    method.to_openff_method(),
                    use_conformers=[conformer],
                )
                bond_order_sets[method] = WibergBondOrderRecord(
                    method=method,
                    values=[
                        WibergBondOrder.from_openff(bond) for bond in molecule.bonds
                    ],
                )

            conformer_records.append(
                ConformerRecord(
                    coordinates=get_coordinates_in_angstrom(conformer),
                    partial_charges=charge_sets,
                    bond_orders=bond_order_sets,
                )
            )

        return cls(
            mapped_smiles=openff_molecule.to_smiles(mapped=True, isomeric=True),
            conformers=conformer_records,
        )

    def to_openff(
        self,
        partial_charge_method: Optional[ChargeMethod] = None,
        bond_order_method: Optional[WibergBondOrderMethod] = None,
    ) -> "openff.toolkit.topology.Molecule":
        """Convert the record to an OpenFF molecule with averaged properties"""

        from openff.toolkit.topology.molecule import Molecule
        from openff.toolkit.topology.molecule import unit as off_unit

        offmol = Molecule.from_mapped_smiles(
            self.mapped_smiles, allow_undefined_stereo=True
        )
        offmol._conformers = [
            conformer.coordinates * off_unit.angstrom
            for conformer in self.conformers
        ]
        if partial_charge_method:
            charges = self.average_partial_charges(partial_charge_method)
            offmol.partial_charges = np.array(
                charges) * off_unit.elementary_charge

        if bond_order_method:
            bond_orders = self.average_bond_orders(bond_order_method)
            for bond in offmol.bonds:
                key = tuple(sorted([bond.atom1_index, bond.atom2_index]))
                bond.fractional_bond_order = bond_orders[key]

        offmol.properties[self._partial_charge_method_name] = partial_charge_method
        offmol.properties[self._bond_order_method_name] = bond_order_method

        return offmol

    def get_partial_charges(self, charge_model: ChargeMethod) -> Array[float]:
        """Get the partial charges for a given charge model

        Parameters
        ----------
        charge_model
            The charge model to get the charges for

        Returns
        -------
        charges: numpy.ndarray
            The partial charges, in units of elementary charge.
            This is a 2D array with shape (n_conformers, n_atoms)
        """
        charge_model = ChargeMethod(charge_model)
        return np.array(
            [
                conformer.partial_charges[charge_model].values
                for conformer in self.conformers
                if charge_model in conformer.partial_charges
            ]
        )

    def get_bond_orders(
        self, bond_order_method: WibergBondOrderMethod
    ) -> Dict[Tuple[int, int], List[float]]:
        """Get the bond orders for a given bond order model

        Parameters
        ----------
        bond_order_method
            The bond order model to get the bond orders for

        Returns
        -------
        bond_orders: Dict[Tuple[int, int], List[float]]
            The fractional bond orders for each bond
        """
        bond_order_method = WibergBondOrderMethod(bond_order_method)
        all_bond_orders = [
            wbo
            for conformer in self.conformers
            if bond_order_method in conformer.bond_orders
            for wbo in conformer.bond_orders[bond_order_method].values
        ]

        orders = defaultdict(list)
        for wbo in all_bond_orders:
            orders[wbo.sorted_atom_indices].append(wbo.bond_order)
        return {k: orders[k] for k in sorted(orders)}

    def average_bond_orders(
        self, bond_order_method: WibergBondOrderMethod
    ) -> Dict[Tuple[int, int], float]:
        """Get the average bond orders for a given bond order model

        Parameters
        ----------
        bond_order_method
            The bond order model to get the bond orders for

        Returns
        -------
        bond_orders: Dict[Tuple[int, int], float]
            The average fractional bond orders for each bond
        """
        bond_orders = self.get_bond_orders(bond_order_method)
        return {k: np.mean(v) for k, v in bond_orders.items()}

    def average_partial_charges(self, charge_model: ChargeMethod) -> Array[float]:
        """Get the average partial charges for a given charge model

        Parameters
        ----------
        charge_model
            The charge model to get the charges for

        Returns
        -------
        charges: numpy.ndarray
            The average partial charges, in units of elementary charge.
            This is a 1D array with shape (n_atoms,)
        """
        return np.mean(self.get_partial_charges(charge_model), axis=0)

    def reorder(self, target_mapped_smiles: str) -> "MoleculeRecord":
        """Reorder the molecule to match the target mapped SMILES"""
        if self.mapped_smiles == target_mapped_smiles:
            return self

        atom_map = map_indexed_smiles(target_mapped_smiles, self.mapped_smiles)
        self_to_target = {k: atom_map[k] for k in sorted(atom_map)}

        return type(self)(
            mapped_smiles=target_mapped_smiles,
            conformers=[
                conformer.map_to(self_to_target) for conformer in self.conformers
            ],
        )
