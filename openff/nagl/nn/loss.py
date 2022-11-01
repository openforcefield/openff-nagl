import abc
import functools
from typing import TYPE_CHECKING, Optional, Union

from openff.nagl.base.metaregistry import create_registry_metaclass

if TYPE_CHECKING:
    import torch
    from openff.nagl.dgl.batch import DGLMoleculeBatch
    from openff.nagl.dgl.molecule import DGLMolecule


class LossFunctionMeta(abc.ABCMeta, create_registry_metaclass("name")):
    pass


class BaseLossFunction(abc.ABC, metaclass=LossFunctionMeta):
    def __call__(
        self,
        predicted_values: "torch.Tensor",
        expected_values: "torch.Tensor",
        molecule: Optional[Union[DGLMolecule, DGLMoleculeBatch]] = None,
    ) -> "torch.Tensor":
        return self.compute(predicted_values, expected_values, molecule)

    @abc.abstractmethod
    def compute(
        self,
        predicted_values: "torch.Tensor",
        expected_values: "torch.Tensor",
        molecule: Optional[Union[DGLMolecule, DGLMoleculeBatch]] = None,
    ) -> "torch.Tensor":
        raise NotImplementedError


class LossMSE(BaseLossFunction):
    name = "mse"

    def compute(
        self,
        predicted_values: "torch.Tensor",
        expected_values: "torch.Tensor",
        molecule: Optional[Union[DGLMolecule, DGLMoleculeBatch]] = None,
    ) -> "torch.Tensor":
        import torch

        mse = torch.nn.functional.mse_loss(predicted_values, expected_values)
        return mse


class LossRMSE(BaseLossFunction):
    name = "rmse"

    def compute(
        self,
        predicted_values: "torch.Tensor",
        expected_values: "torch.Tensor",
        molecule: Optional[Union[DGLMolecule, DGLMoleculeBatch]] = None,
    ) -> "torch.Tensor":
        import torch

        mse = torch.nn.functional.mse_loss(predicted_values, expected_values)
        return torch.sqrt(mse)


class LossDipoleMSE(BaseLossFunction):
    name = "dipole_mse"

    def compute(
        self,
        predicted_values: "torch.Tensor",
        expected_values: "torch.Tensor",
        molecule: Optional[Union[DGLMolecule, DGLMoleculeBatch]] = None,
    ) -> "torch.Tensor":
        import torch

        from openff.nagl.metrics.properties import calculate_dipole_in_angstrom

        conformers = torch.transpose(molecule.graph.ndata["coordinates"], 0, 1)
        loss = torch.zeros(1).type_as(predicted_values)
        for conf in conformers:
            predicted_dipole = calculate_dipole_in_angstrom(conf, predicted_values)
            expected_dipole = calculate_dipole_in_angstrom(conf, expected_values)
            mse = torch.nn.functional.mse_loss(predicted_dipole, expected_dipole)
            loss += mse
        loss /= len(conformers)
        return loss


class LossDipoleRMSE(BaseLossFunction):
    name = "dipole_rmse"

    def compute(
        self,
        predicted_values: "torch.Tensor",
        expected_values: "torch.Tensor",
        molecule: Optional[Union[DGLMolecule, DGLMoleculeBatch]] = None,
    ) -> "torch.Tensor":
        import torch

        from openff.nagl.metrics.properties import calculate_dipole_in_angstrom

        conformers = torch.transpose(molecule.graph.ndata["coordinates"], 0, 1)
        loss = torch.zeros(1).type_as(predicted_values)
        for conf in conformers:
            predicted_dipole = calculate_dipole_in_angstrom(conf, predicted_values)
            expected_dipole = calculate_dipole_in_angstrom(conf, expected_values)
            mse = torch.nn.functional.mse_loss(predicted_dipole, expected_dipole)
            loss += torch.sqrt(mse)
        loss /= len(conformers)
        return loss
