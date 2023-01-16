import dgl
import torch

from ._base import ActivationFunction, BaseGCNStack


class GINConv(torch.nn.Module):

    def __init__(
        self,
        n_input_features: int,
        n_output_features: int,
        aggregator_type: str,
        dropout: float,
        activation_function: ActivationFunction,
        init_eps: float = 0.0,
        learn_eps: bool = False
    ):
        super().__init__()

        # self.activation = activation_function
        self.feat_drop = torch.nn.Dropout(dropout)
        self.gcn = dgl.nn.pytorch.GINConv(
            apply_func=torch.nn.Linear(n_input_features, n_output_features),
            aggregator_type=aggregator_type,
            init_eps=init_eps,
            learn_eps=learn_eps,
            activation=activation_function
        )

    def reset_parameters(self):
        pass
        # self.gcn.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, inputs: torch.Tensor):
        dropped_inputs = self.feat_drop(inputs)
        output = self.gcn(graph, dropped_inputs)
        return output

    @property
    def activation(self):
        return self.gcn.activation

    @property
    def fc_self(self):
        return self.gcn.apply_func


class GINConvStack(BaseGCNStack[GINConv]):
    """A wrapper around a stack of GINConv graph convolutional layers"""

    name = "GINConv"
    available_aggregator_types = ["sum", "max", "mean"]
    default_aggregator_type = "sum"
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
        init_eps: float = 0.0,
        learn_eps: bool = False,
        **kwargs,
    ) -> GINConv:

        return GINConv(
            n_input_features=n_input_features,
            n_output_features=n_output_features,
            aggregator_type=aggregator_type,
            dropout=dropout,
            activation_function=activation_function,
            init_eps=init_eps,
            learn_eps=learn_eps
        )
