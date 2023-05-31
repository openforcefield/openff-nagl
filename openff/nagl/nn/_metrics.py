import abc
import functools
import typing
from pydantic.main import ModelMetaclass

import torch

from openff.nagl._base.metaregistry import create_registry_metaclass
from openff.nagl._base.base import ImmutableModel

if typing.TYPE_CHECKING:
    import torch
    from openff.nagl.molecule._dgl.batch import DGLMoleculeBatch
    from openff.nagl.molecule._dgl.molecule import DGLMolecule


# class MetricMeta(ModelMetaclass, abc.ABCMeta, create_registry_metaclass("name")):
#     pass

class BaseMetric(ImmutableModel, abc.ABC):
    name: typing.Literal[""]
    def __call__(
        self,
        predicted_values: "torch.Tensor",
        expected_values: "torch.Tensor"
    ) -> "torch.Tensor":
        return self.compute(predicted_values, expected_values)

    @abc.abstractmethod
    def compute(
        self,
        predicted_values: "torch.Tensor",
        expected_values: "torch.Tensor"
    ) -> "torch.Tensor":
        raise NotImplementedError


class RMSEMetric(BaseMetric):
    name: typing.Literal["rmse"] = "rmse"

    def compute(self, predicted_values, expected_values):
        return torch.sqrt(torch.mean((predicted_values - expected_values) ** 2))


class MSEMetric(BaseMetric):
    name: typing.Literal["mse"] = "mse"

    def compute(self, predicted_values, expected_values):
        return torch.mean((predicted_values - expected_values) ** 2)


class MAEMetric(BaseMetric):
    name: typing.Literal["mae"] = "mae"

    def compute(self, predicted_values, expected_values):
        return torch.mean(torch.abs(predicted_values - expected_values))


MetricType = typing.Union[RMSEMetric, MSEMetric, MAEMetric]