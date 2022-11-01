import abc
from typing import ClassVar, Dict, Generic, List, Optional, Type, TypeVar

import dgl
import dgl.nn.pytorch
import torch.nn
import torch.nn.functional

from openff.nagl.nn.activation import ActivationFunction
from openff.nagl.nn.base import ContainsLayersMixin
from openff.nagl.base.metaregistry import create_registry_metaclass

GCNLayerType = TypeVar("GCNLayerType", bound=torch.nn.Module)


# class GCNStackMeta(abc.ABCMeta):
#     """A metaclass for GCN stacks.

#     This metaclass is used to register GCN layers by name.
#     """

#     registry: ClassVar[Dict[str, Type]] = {}

#     def __init__(cls, name, bases, namespace, **kwargs):
#         super().__init__(name, bases, namespace, **kwargs)
#         if hasattr(cls, "name") and cls.name:
#             cls.registry[cls.name] = cls

#     @classmethod
#     def get_gcn_class(cls, class_name: str):
#         if isinstance(class_name, cls):
#             return class_name
#         if isinstance(type(class_name), cls):
#             return type(class_name)
#         try:
#             return cls.registry[class_name]
#         except KeyError:
#             raise ValueError(
#                 f"Unknown GCN layer type: {class_name}. "
#                 f"Supported types: {list(cls.registry.keys())}"
#             )


class GCNStackMeta(abc.ABCMeta, create_registry_metaclass("name")):
    pass

class BaseGCNStack(
    torch.nn.ModuleList,
    Generic[GCNLayerType],
    ContainsLayersMixin,
    abc.ABC,
    metaclass=GCNStackMeta,
):
    """A wrapper around a stack of GCN graph convolutional layers.

    Note:
        This class is based on the ``dgllife.model.SAGEConv`` module.
    """

    hidden_feature_sizes: List[GCNLayerType]

    @property
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        pass

    @property
    @classmethod
    @abc.abstractmethod
    def available_aggregator_types(cls) -> str:
        """The aggregator options to use for the GCN layers."""

    @property
    @classmethod
    @abc.abstractmethod
    def default_aggregator_type(cls) -> str:
        """The aggregator options to use for the GCN layers."""

    @property
    @classmethod
    @abc.abstractmethod
    def default_dropout(cls) -> str:
        """The aggregator options to use for the GCN layers."""

    @property
    @classmethod
    @abc.abstractmethod
    def default_activation_function(cls) -> str:
        """The aggregator options to use for the GCN layers."""

    @classmethod
    def _check_input_lengths(
        cls,
        n_layers: int,
        layer_activation_functions: Optional[List[ActivationFunction]] = None,
        layer_dropout: Optional[List[float]] = None,
        layer_aggregator_types: Optional[List[str]] = None,
    ):
        layer_activation_functions, layer_dropout = super()._check_input_lengths(
            n_layers,
            layer_activation_functions,
            layer_dropout,
        )

        if layer_aggregator_types is None:
            layer_aggregator_types = cls.default_aggregator_type
        layer_aggregator_types = cls._check_argument_input_length(
            n_layers,
            layer_aggregator_types,
            "layer_aggregator_types",
        )
        return layer_activation_functions, layer_dropout, layer_aggregator_types

    @classmethod
    def with_layers(
        cls,
        n_input_features: int,
        hidden_feature_sizes: List[int],
        layer_activation_functions: Optional[List[ActivationFunction]] = None,
        layer_dropout: Optional[List[float]] = None,
        layer_aggregator_types: Optional[List[str]] = None,
    ):
        """Create this model with layers with the specified parameters."""
        obj = cls()

        n_layers = len(hidden_feature_sizes)
        (
            layer_activation_functions,
            layer_dropout,
            layer_aggregator_types,
        ) = cls._check_input_lengths(
            n_layers,
            layer_activation_functions,
            layer_dropout,
            layer_aggregator_types,
        )

        for i in range(n_layers):
            n_output_features = hidden_feature_sizes[i]
            activation_function = layer_activation_functions[i]
            dropout = layer_dropout[i]
            aggregator_type = layer_aggregator_types[i]

            obj.append_gcn_layer(
                n_input_features=n_input_features,
                n_output_features=n_output_features,
                activation_function=activation_function,
                aggregator_type=aggregator_type,
                dropout=dropout,
            )
            n_input_features = n_output_features
        return obj

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_feature_sizes = []

    def append_gcn_layer(
        self,
        n_output_features: int,
        n_input_features: Optional[int] = None,
        aggregator_type: Optional[str] = None,
        dropout: Optional[float] = None,
        activation_function: Optional[ActivationFunction] = None,
    ):
        """Add a new layer to the stack."""
        if n_input_features is None:
            try:
                n_input_features = self.hidden_feature_sizes[-1]
            except IndexError:
                raise ValueError(
                    "Must specify n_input_features if no layers have been created yet."
                )

        self.hidden_feature_sizes.append(n_output_features)
        self.append(
            self.create_gcn_layer(
                n_input_features,
                n_output_features,
                aggregator_type,
                dropout,
                activation_function,
            )
        )

    @classmethod
    def create_gcn_layer(
        cls,
        n_input_features: int,
        n_output_features: int,
        aggregator_type: Optional[str] = None,
        dropout: Optional[float] = None,
        activation_function: Optional[ActivationFunction] = None,
        **kwargs,
    ) -> GCNLayerType:
        """Create a new GCN layer."""
        if aggregator_type is None:
            aggregator_type = cls.default_aggregator_type
        if dropout is None:
            dropout = cls.default_dropout
        if activation_function is None:
            activation_function = cls.default_activation_function
        activation = ActivationFunction.get_value(activation_function)
        # activation = ActivationFunction.get_function(activation_function)
        return cls._create_gcn_layer(
            n_input_features=n_input_features,
            n_output_features=n_output_features,
            aggregator_type=aggregator_type,
            dropout=dropout,
            activation_function=activation,
            **kwargs,
        )

    @classmethod
    @abc.abstractmethod
    def _create_gcn_layer(
        cls,
        n_input_features: int,
        n_output_features: int,
        aggregator_type: str,
        dropout: float,
        activation_function: ActivationFunction,
        **kwargs,
    ) -> GCNLayerType:
        """A function which returns an instantiated GCN layer.

        Args:
            in_feats: Number of input node features.
            out_feats: Number of output node features.
            activation_function: The activation_function function to.
            dropout: `The dropout probability.
            aggregator_type: The aggregator type, which can be one of ``"sum"``,
                ``"max"``, ``"mean"``.
            init_eps: The initial value of epsilon.
            learn_eps: If True epsilon will be a learnable parameter.

        Returns:
            The instantiated GCN layer.
        """

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self:
            gnn.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, inputs: torch.Tensor) -> torch.Tensor:
        """Update node representations.

        Args:
            graph: The batch of graphs to operate on.
            inputs: The inputs to the layers with shape=(n_nodes, in_feats).

        Returns
            The output hidden features with shape=(n_nodes, hidden_feats[-1]).
        """
        for gnn in self:
            inputs = gnn(graph, inputs)
        return inputs
