import abc
import copy
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule

    from openff.nagl.storage.record import (
        ChargeMethod,
        WibergBondOrderMethod,
    )


class LabelFunction(abc.ABC):
    def __call__(self, molecule: "OFFMolecule") -> Dict[str, torch.Tensor]:
        return self.run(molecule)

    @abc.abstractmethod
    def run(self, molecule: "OFFMolecule") -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class EmptyLabeller(LabelFunction):
    def run(self, molecule: "OFFMolecule") -> Dict[str, torch.Tensor]:
        return {}


class LabelPrecomputedMolecule(LabelFunction):
    def __init__(
        self,
        partial_charge_method: Optional["ChargeMethod"] = None,
        bond_order_method: Optional["WibergBondOrderMethod"] = None,
    ):
        from openff.nagl.storage.record import (
            ChargeMethod,
            WibergBondOrderMethod,
        )

        self.partial_charge_label = None
        self.bond_order_label = None
        self.partial_charge_method = None
        self.bond_order_method = None


        if partial_charge_method is not None:
            self.partial_charge_method = ChargeMethod(partial_charge_method)
            self.partial_charge_label = f"{self.partial_charge_method}-charges"
        if bond_order_method is not None:
            self.bond_order_method = WibergBondOrderMethod(bond_order_method)
            self.bond_order_label = f"{self.bond_order_label}-wbo"


    def run(self, molecule: "OFFMolecule") -> Dict[str, torch.Tensor]:
        from openff.nagl.utils.openff import get_unitless_charge

        labels = {}
        if self.partial_charge_method is not None:
            charges = [
                get_unitless_charge(atom.partial_charge) for atom in molecule.atoms
            ]
            labels[self.partial_charge_label] = torch.tensor(charges, dtype=torch.float)

        if self.bond_order_method is not None:
            orders = [bond.fractional_bond_order for bond in molecule.bonds]
            labels[self.bond_order_label] = torch.tensor(orders, dtype=torch.float)

        return labels


class ComputeAndLabelMolecule(LabelFunction):
    def __init__(
        self,
        partial_charge_methods: Tuple["ChargeMethod"] = tuple(),
        bond_order_methods: Tuple["WibergBondOrderMethod"] = tuple(),
        n_conformers: int = 500,
        rms_cutoff: float = 0.05,
    ):
        from openff.nagl.storage.record import (
            ChargeMethod,
            WibergBondOrderMethod,
        )

        self.n_conformers = n_conformers
        self.rms_cutoff = rms_cutoff
        self.partial_charge_methods = [
            ChargeMethod(method) for method in partial_charge_methods
        ]
        self.bond_order_methods = [
            WibergBondOrderMethod(method) for method in bond_order_methods
        ]

    def run(self, molecule: "OFFMolecule") -> Dict[str, torch.Tensor]:
        from openff.toolkit.topology.molecule import Molecule as OFFMolecule
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

        molecule = copy.deepcopy(molecule)
        molecule.generate_conformers(
            n_conformers=self.n_conformers,
            rms_cutoff=self.rms_cutoff * off_unit.angstrom,
        )
        molecule.apply_elf_conformer_selection()

        conformer_records = []
        for conformer in molecule.conformers:
            charge_sets = {}
            for method in self.partial_charge_methods:
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
            for method in self.bond_order_methods:
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


LabelFunctionLike = Union[
    LabelFunction, Callable[["OFFMolecule"], Dict[str, torch.Tensor]]
]
