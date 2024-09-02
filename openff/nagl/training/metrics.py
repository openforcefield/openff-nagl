"""
Metrics for evaluating loss
"""
import abc
import typing

import torch

from openff.nagl._base.base import ImmutableModel


if typing.TYPE_CHECKING:
    import torch


class BaseMetric(ImmutableModel, abc.ABC):
    """
    Base class for metrics to evaluate loss between predicted and expected values.
    """
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


class MSEMetric(BaseMetric):
    name: typing.Literal["mse"] = "mse"

    def compute(self, predicted_values, expected_values):
        loss = torch.nn.MSELoss()
        return loss(predicted_values, expected_values)


class MAEMetric(BaseMetric):
    name: typing.Literal["mae"] = "mae"

    def compute(self, predicted_values, expected_values):
        loss = torch.nn.L1Loss()
        return loss(predicted_values, expected_values)


MetricType = typing.Union[RMSEMetric, MSEMetric, MAEMetric]

METRICS = {
    "rmse": RMSEMetric,
    "mse": MSEMetric,
    "mae": MAEMetric
}
"""
Mapping from metric names to the corresponding classes.
"""


def get_metric_type(metric: typing.Union[MetricType, str]) -> MetricType:
    """
    Get the metric class instance from a string or class.
    """
    if isinstance(metric, BaseMetric):
        return metric
    elif isinstance(metric, str):
        metric = metric.lower()
        return METRICS[metric]()
    else:
        raise ValueError(f"Unknown metric type: {metric}")
