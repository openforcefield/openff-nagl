import abc
from typing import TYPE_CHECKING, Dict, Callable, Union, Optional

import torch

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule
    from gnn_charge_models.storage.record import ChargeMethod, WibergBondOrderMethod


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
    def __init__(self,
                 partial_charge_method: Optional["ChargeMethod"] = None,
                 bond_order_method: Optional["WibergBondOrderMethod"] = None,
                 ) -> Dict[str, torch.Tensor]:
        from gnn_charge_models.storage.record import ChargeMethod, WibergBondOrderMethod

        if partial_charge_method is not None:
            partial_charge_method = ChargeMethod(partial_charge_method)
        if bond_order_method is not None:
            bond_order_method = WibergBondOrderMethod(bond_order_method)

        self.partial_charge_method = partial_charge_method
        self.bond_order_method = bond_order_method

    def run(self, molecule: "OFFMolecule") -> Dict[str, torch.Tensor]:
        from gnn_charge_models.utils.openff import get_unitless_charge
        labels = {}
        if self.partial_charge_method is not None:
            charge_label = f"{self.partial_charge_method.value}-charges"
            charges = [
                get_unitless_charge(atom.partial_charge)
                for atom in molecule.atoms
            ]
            labels[charge_label] = torch.tensor(charges, dtype=torch.float)

        if self.bond_order_method is not None:
            bond_order_label = f"{self.bond_order_method.value}-wbo"
            orders = [bond.fractional_bond_order for bond in molecule.bonds]
            labels[bond_order_label] = torch.tensor(orders, dtype=torch.float)

        return labels


LabelFunctionLike = Union[
    LabelFunction,
    Callable[
        ["OFFMolecule"],
        Dict[str, torch.Tensor]
    ]
]
