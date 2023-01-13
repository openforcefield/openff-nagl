import abc
import functools
from typing import TYPE_CHECKING, Optional, Union

from openff.nagl._base.metaregistry import create_registry_metaclass

if TYPE_CHECKING:
    import torch
    from openff.nagl._dgl.batch import DGLMoleculeBatch
    from openff.nagl._dgl.molecule import DGLMolecule


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

