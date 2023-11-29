import abc
import typing

import torch

from openff.nagl._base.metaregistry import create_registry_metaclass
from openff.nagl._base.base import ImmutableModel

try:
    from pydantic.v1.main import ModelMetaclass
except ImportError:
    from pydantic.main import ModelMetaclass

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
        loss = torch.nn.MSELoss()
        return torch.sqrt(loss(predicted_values, expected_values))
        # return torch.sqrt(torch.mean((predicted_values - expected_values) ** 2))


class WeightedRMSEMetric(BaseMetric):
    name: typing.Literal["weighted_rmse"] = "weighted_rmse"

    def compute(self, predicted_values, expected_values):
        difference = predicted_values - expected_values
        weights = torch.abs(expected_values)
        mse = torch.mean(weights * difference ** 2)
        return torch.sqrt(mse)

class MSEMetric(BaseMetric):
    name: typing.Literal["mse"] = "mse"

    def compute(self, predicted_values, expected_values):
        loss = torch.nn.MSELoss()
        return loss(predicted_values, expected_values)
        # return torch.mean((predicted_values - expected_values) ** 2)


class MAEMetric(BaseMetric):
    name: typing.Literal["mae"] = "mae"

    def compute(self, predicted_values, expected_values):
        loss = torch.nn.L1Loss()
        return loss(predicted_values, expected_values)
        # return torch.mean(torch.abs(predicted_values - expected_values))


MetricType = typing.Union[RMSEMetric, WeightedRMSEMetric, MSEMetric, MAEMetric]

METRICS = {
    "rmse": RMSEMetric,
    "weighted_rmse": WeightedRMSEMetric,
    "mse": MSEMetric,
    "mae": MAEMetric
}


def get_metric_type(metric):
    if isinstance(metric, BaseMetric):
        return metric
    elif isinstance(metric, str):
        metric = metric.lower()
        return METRICS[metric]()