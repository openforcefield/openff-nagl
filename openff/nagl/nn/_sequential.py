from typing import ClassVar, List, Optional

import torch.nn

from ._base import ContainsLayersMixin
from .activation import ActivationFunction


class SequentialLayers(torch.nn.Sequential, ContainsLayersMixin):
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
        layer_activation_functions, layer_dropout = cls._check_input_lengths(
            n_layers,
            layer_activation_functions,
            layer_dropout,
        )

        hidden_feature_sizes = [n_input_features, *hidden_feature_sizes]
        layers = []

        for i in range(n_layers):
            linear = torch.nn.Linear(
                hidden_feature_sizes[i], hidden_feature_sizes[i + 1]
            )
            activation = ActivationFunction.get_value(layer_activation_functions[i])
            dropout = torch.nn.Dropout(p=layer_dropout[i])
            layers.extend([linear, activation, dropout])

        return cls(*layers)

    def copy(self, copy_weights: bool = False):
        layers = []
        for layer in self:
            if isinstance(layer, torch.nn.Linear):
                layers.append(torch.nn.Linear(layer.in_features, layer.out_features))
            elif isinstance(layer, ActivationFunction):
                layers.append(layer)
            elif isinstance(layer, torch.nn.Dropout):
                layers.append(torch.nn.Dropout(p=layer.p))
            else:
                raise NotImplementedError()

        assert len(layers) == len(self)
        copied = type(self)(*layers)

        if copy_weights:
            copied.load_state_dict(self.state_dict())
        return copied
