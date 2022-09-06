import dgl
import torch

from .base import BaseGCNStack, ActivationFunction


class SAGEConvStack(BaseGCNStack[dgl.nn.pytorch.SAGEConv]):
    """A wrapper around a stack of SAGEConv graph convolutional layers"""

    layer_type = "SAGEConv"
    available_aggregator_types = ["mean", "gcn", "pool", "lstm"]
    default_aggregator_type = "mean"
    default_dropout = 0.0
    default_activation_function = ActivationFunction.ReLU

    @classmethod
    def _create_gcn_layer(
        cls,
        n_input_features: int,
        n_output_features: int,
        aggregator_type: str,
        dropout: float,
        activation_function: ActivationFunction,
        **kwargs,
    ) -> dgl.nn.pytorch.SAGEConv:

        return dgl.nn.pytorch.SAGEConv(
            in_feats=n_input_features,
            out_feats=n_output_features,
            activation=activation_function,
            feat_drop=dropout,
            aggregator_type=aggregator_type,

        )
