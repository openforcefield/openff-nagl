import copy
import enum
from collections import defaultdict

from typing import List, Tuple, Dict, Optional, NamedTuple

from gnn_charge_models.base.array import Array
from gnn_charge_models.base.quantity import Quantity

import numpy as np
from openff.units import unit as openff_unit
from pydantic import validator

from gnn_charge_models.base.base import ImmutableModel
from gnn_charge_models.utils.openff import map_indexed_smiles


class Record(ImmutableModel):
    class Config(ImmutableModel.Config):
        orm_mode = True


class ChargeMethod(enum.Enum):
    """
    The method used to calculate the partial charges.
    """
    AM1 = "am1"
    AM1BCC = "am1bcc"

    def to_openff_method(self) -> str:
        options = {"am1": "am1-mulliken", "am1bcc": "am1bcc"}
        return options[self.value]


class WibergBondOrderMethod(enum.Enum):
    AM1 = "am1"

    def to_openff_method(self) -> str:
        options = {"am1": "am1-wiberg"}
        return options[self.value]


class PartialChargeRecord(Record):
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


# class WibergBondOrder(Record):
#     atom_index_1: int
#     atom_index_2: int
#     bond_order: float

#     @classmethod
#     def from_openff(cls, openff_bond):
#         return cls(
#             atom_index_1=openff_bond.atom1_index,
#             atom_index_2=openff_bond.atom2_index,
#             bond_order=openff_bond.fractional_bond_order,
#         )

#     def map_to(self, mapping: Dict[int, int]):
#         return type(self)(
#             atom_index_1=mapping[self.atom_index_1],
#             atom_index_2=mapping[self.atom_index_2],
#             bond_order=self.bond_order,
#         )

#     @property
#     def sorted_atom_indices(self):
#         return tuple(sorted((self.atom_index_1, self.atom_index_2)))


class WibergBondOrder(NamedTuple):
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
    method: WibergBondOrderMethod = WibergBondOrderMethod.AM1
    values: List[WibergBondOrder]

    def map_to(self, mapping: Dict[int, int]):
        return type(self)(
            method=self.method,
            values=[
                bond.map_to(mapping) for bond in self.values
            ],
        )


class ConformerRecord(Record):
    coordinates: Array[float]
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
        v = v.reshape((-1, 3))
        v.flags.writeable = False
        return v

    @validator("partial_charges", "bond_orders", pre=True)
    def _validate_unique_methods(cls, v):
        if not isinstance(v, dict):
            new_v = {}
            for v_ in v:
                new_v[v_.method] = v_
            return new_v
        return v

    def map_to(self, mapping: Dict[int, int]):
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
    mapped_smiles: str
    conformers: List[ConformerRecord]

    @property
    def smiles(self):
        return self.mapped_smiles

    @classmethod
    def from_openff(
        cls,
        openff_molecule,
        charge_methods: List[ChargeMethod] = [],
        bond_order_methods: List[WibergBondOrderMethod] = [],
        n_conformer_pool: int = 500,
        n_conformers: int = 10,
        rms_cutoff: float = 0.05
    ):
        from openff.toolkit.topology.molecule import unit as toolkit_unit
        openff_molecule = copy.deepcopy(openff_molecule)
        openff_molecule.generate_conformers(
            n_conformers=n_conformer_pool,
            rms_cutoff=rms_cutoff * toolkit_unit.angstrom,
        )
        openff_molecule.apply_elf_conformer_selection(limit=n_conformers)

        conformers = []
        for conformer in openff_molecule.conformers:
            charges = {}
            for method in charge_methods:
                openff_molecule.assign_partial_charges(
                    method=method.to_openff_method(),
                    use_conformers=[conformer]
                )
                charges[method] = PartialChargeRecord(
                    method=method,
                    values=conformer.partial_charges,
                )

            bond_orders = {}
            for method in bond_order_methods:
                openff_molecule.assign_fractional_bond_orders(
                    method=method.to_openff_method(),
                    use_conformers=[conformer]
                )
                bond_orders[method] = WibergBondOrderRecord(
                    method=method,
                    values=[
                        WibergBondOrder.from_openff(bond)
                        for bond in openff_molecule.bonds
                    ],
                )
            conformer_record = ConformerRecord(
                coordinates=conformer,
                partial_charges=charges,
                bond_orders=bond_orders,
            )
            conformers.append(conformer_record)

        return cls(
            mapped_smiles=openff_molecule.to_smiles(mapped=True, isomeric=True),
            conformers=conformers,
        )

    def to_openff(self, partial_charge_method: Optional[ChargeMethod] = None, bond_order_method: Optional[WibergBondOrderMethod] = None):
        from openff.toolkit.topology.molecule import Molecule, unit as off_unit

        print(self.smiles)

        offmol = Molecule.from_mapped_smiles(
            self.mapped_smiles, allow_undefined_stereo=True)
        if partial_charge_method:
            charges = self.average_partial_charges(partial_charge_method)
            offmol.partial_charges = np.array(
                charges) * off_unit.elementary_charge

        if bond_order_method:
            bond_orders = self.average_bond_orders(bond_order_method)
            for bond in offmol.bonds:
                key = tuple(sorted([bond.atom1_index, bond.atom2_index]))
                bond.fractional_bond_order = bond_orders[key]

        return offmol

    def get_partial_charges(self, charge_model: ChargeMethod) -> Array[float]:
        charge_model = ChargeMethod(charge_model)
        return np.array([
            conformer.partial_charges[charge_model].values
            for conformer in self.conformers
            if charge_model in conformer.partial_charges
        ])

    def get_bond_orders(self, bond_order_method: WibergBondOrderMethod) -> Dict[Tuple[int, int], List[float]]:
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
        return {
            k: orders[k]
            for k in sorted(orders)
        }

    def average_bond_orders(self, bond_order_method: WibergBondOrderMethod) -> Dict[Tuple[int, int], float]:
        bond_orders = self.get_bond_orders(bond_order_method)
        print(bond_orders)
        return {
            k: np.mean(v)
            for k, v in bond_orders.items()
        }

    def average_partial_charges(self, charge_model: ChargeMethod) -> Array[float]:
        return np.mean(self.get_partial_charges(charge_model), axis=0)

    def reorder(self, target_mapped_smiles: str) -> "MoleculeRecord":
        if self.mapped_smiles == target_mapped_smiles:
            return self

        atom_map = map_indexed_smiles(target_mapped_smiles, self.mapped_smiles)
        self_to_target = {k: atom_map[k] for k in sorted(atom_map)}

        return type(self)(
            mapped_smiles=target_mapped_smiles,
            conformers=[
                conformer.map_to(self_to_target)
                for conformer in self.conformers
            ],
        )
