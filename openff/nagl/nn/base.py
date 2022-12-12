from typing import Any, List, Optional

from openff.nagl.utils.utils import is_iterable

from .activation import ActivationFunction

__all__ = [
    "ContainsLayersMixin",
]


class ContainsLayersMixin:
    @classmethod
    def _check_input_lengths(
        cls,
        n_layers: int,
        layer_activation_functions: Optional[List[ActivationFunction]] = None,
        layer_dropout: Optional[List[float]] = None,
    ):
        if layer_activation_functions is None:
            layer_activation_functions = cls.default_activation_function
        if layer_dropout is None:
            layer_dropout = cls.default_dropout

        layer_activation_functions = cls._check_argument_input_length(
            n_layers,
            layer_activation_functions,
            "layer_activation_functions",
        )
        layer_dropout = cls._check_argument_input_length(
            n_layers,
            layer_dropout,
            "layer_dropout",
        )
        return layer_activation_functions, layer_dropout

    @staticmethod
    def _check_argument_input_length(n_layers: int, obj: Any, argname: str):
        if not is_iterable(obj):
            obj = [obj] * n_layers
        if not len(obj) == n_layers:
            raise ValueError(
                f"`{argname}` (length {len(obj)}) must be a list of "
                f"same length as `hidden_feature_sizes` (length {n_layers})."
            ) from None
        return obj
