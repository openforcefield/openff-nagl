from typing import ClassVar, List, Optional

import torch.nn
from .activation import ActivationFunction
from gnn_charge_models.utils.utils import assert_same_lengths


class SequentialLayers(torch.nn.Sequential):

    default_activation_function: ClassVar[ActivationFunction] = ActivationFunction.ReLU
    default_dropout: ClassVar[float] = 0.0

    @classmethod
    def with_layers(
        cls,
        n_input_features: int,
        hidden_feature_sizes: List[int],
        layer_activation_functions: Optional[List[ActivationFunction]] = None,
        layer_dropout: Optional[List[float]] = None,
    ):

        n_layers = len(hidden_feature_sizes)

        if layer_activation_functions is None:
            layer_activation_functions = [
                cls.default_activation_function] * n_layers
        if layer_dropout is None:
            layer_dropout = [cls.default_dropout] * n_layers

        try:
            assert_same_lengths(hidden_feature_sizes,
                                layer_activation_functions, layer_dropout)
        except AssertionError as e:
            err = (
                "`hidden_feature_sizes`, `layer_activation_functions`, and `layer_dropout` must be the same length. "
                + e.msg
            )
            raise ValueError(err) from None

        hidden_feature_sizes = [n_input_features, *hidden_feature_sizes]
        layers = []

        for i in range(n_layers):
            linear = torch.nn.Linear(
                hidden_feature_sizes[i], hidden_feature_sizes[i + 1])
            activation = ActivationFunction.get_value(
                layer_activation_functions[i])
            dropout = torch.nn.Dropout(p=layer_dropout[i]),
            layers.extend([linear, activation(), dropout])

        return cls(*layers)
