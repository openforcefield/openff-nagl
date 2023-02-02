import copy
from typing import TYPE_CHECKING, Union

import torch
from openff.utilities import requires_package

from ._base import ActivationFunction, BaseGCNStack, BaseConvModule
import openff.nagl.nn.gcn._function as _fn

if TYPE_CHECKING:
    import dgl


class GINConvLayer(BaseConvModule):
    def __init__(
        self,
        apply_func=None,
        aggregator_type='sum',
        init_eps=0,
        learn_eps=False,
        activation=None
    ):
        super().__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        if aggregator_type not in GINConvStack.available_aggregator_types:
            raise KeyError(
                f'Aggregator type {aggregator_type} not recognized.'
                )
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        raise NotImplementedError
        # TODO: go back and do this

        # _reducer = getattr(_fn, self._aggregator_type)

        # with graph.local_scope():
        #     aggregate_fn = _fn.copy_u('h', 'm')
        #     if edge_weight is not None:
        #         assert edge_weight.shape[0] == graph.number_of_edges()
        #         graph.edata['_edge_weight'] = edge_weight
        #         aggregate_fn = _fn.u_mul_e('h', '_edge_weight', 'm')

        #     feat_src, feat_dst = expand_as_pair(feat, graph)
        #     graph.srcdata['h'] = feat_src
        #     graph.update_all(aggregate_fn, _reducer('m', 'neigh'))
        #     rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
        #     if self.apply_func is not None:
        #         rst = self.apply_func(rst)
        #     # activation
        #     if self.activation is not None:
        #         rst = self.activation(rst)
        #     return rst

class BaseGINConv(torch.nn.Module):
    def reset_parameters(self):
        pass
        # self.gcn.reset_parameters()

    def forward(self, graph: "dgl.DGLGraph", inputs: torch.Tensor):
        dropped_inputs = self.feat_drop(inputs)
        output = self.gcn(graph, dropped_inputs)
        return output

    @property
    def activation(self):
        return self.gcn.activation

    @property
    def fc_self(self):
        return self.gcn.apply_func


class GINConv(BaseGINConv):
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

        self.activation = activation_function
        self.feat_drop = torch.nn.Dropout(dropout)
        self.gcn = GINConvLayer(
            nn=torch.nn.Linear(n_input_features, n_output_features),
            aggregator_type=aggregator_type,
            init_eps=init_eps,
            learn_eps=learn_eps,
            train_eps=True
        )


class DGLGINConv(BaseGINConv):
    @requires_package("dgl")
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
        import dgl

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
    ) -> Union[GINConv, DGLGINConv]:
        try:
            return cls._create_gcn_layer_dgl(
                n_input_features=n_input_features,
                n_output_features=n_output_features,
                aggregator_type=aggregator_type,
                dropout=dropout,
                activation_function=activation_function,
                init_eps=init_eps,
                learn_eps=learn_eps
            )
        except ImportError:
            return cls._create_gcn_layer_nagl(
                n_input_features=n_input_features,
                n_output_features=n_output_features,
                aggregator_type=aggregator_type,
                dropout=dropout,
                activation_function=activation_function,
                init_eps=init_eps,
                learn_eps=learn_eps
            )

    @classmethod
    def _create_gcn_layer_nagl(
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
    
    @classmethod
    def _create_gcn_layer_dgl(
        cls,
        n_input_features: int,
        n_output_features: int,
        aggregator_type: str,
        dropout: float,
        activation_function: ActivationFunction,
        init_eps: float = 0.0,
        learn_eps: bool = False,
        **kwargs,
    ) -> DGLGINConv:
        return DGLGINConv(
            n_input_features=n_input_features,
            n_output_features=n_output_features,
            aggregator_type=aggregator_type,
            dropout=dropout,
            activation_function=activation_function,
            init_eps=init_eps,
            learn_eps=learn_eps
        )
    
    @property
    def _is_dgl(self):
        return not isinstance(self[0].gcn, BaseConvModule)
    
    def _as_nagl(self, copy_weights: bool = False):
        if self._is_dgl:
            new_obj = type(self)()
            new_obj.hidden_feature_sizes = self.hidden_feature_sizes
            for layer in self:
                n_input_features = layer.gcn.apply_func.in_features
                n_output_features = layer.gcn.apply_func.out_features
                aggregator_type = layer.gcn._aggregator_type
                dropout = layer.feat_drop.p
                activation_function = layer.gcn.activation
                learn_eps = isinstance(layer.gcn.eps, torch.nn.Parameter)
                eps = float(layer.gcn.eps.data[0])


                new_layer = self._create_gcn_layer_nagl(
                    n_input_features=n_input_features,
                    n_output_features=n_output_features,
                    aggregator_type=aggregator_type,
                    dropout=dropout,
                    activation_function=activation_function,
                    init_eps=eps,
                    learn_eps=learn_eps
                )
                if copy_weights:
                    new_layer.load_state_dict(layer.state_dict())
                new_obj.append(new_layer)
                
            return copy.deepcopy(new_obj)
        return copy.deepcopy(self)