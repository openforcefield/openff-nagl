import copy
from typing import TYPE_CHECKING, Dict, Literal, Optional, Union

import torch
from openff.utilities import requires_package
from openff.utilities.exceptions import MissingOptionalDependencyError

from openff.nagl.nn.gcn import _function as _fn

from ._base import ActivationFunction, BaseConvModule, BaseGCNStack

if TYPE_CHECKING:
    import dgl


class SAGEConv(BaseConvModule):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        aggregator_type: Literal["mean", "gcn", "pool", "lstm"],
        feat_drop: float,
        bias: bool = True,
        norm: Optional[callable] = None,
        activation: Optional[callable] = None,
    ):
        super().__init__()
        if aggregator_type not in SAGEConvStack.available_aggregator_types:
            raise ValueError(
                f"Aggregator type {aggregator_type} not supported by {SAGEConvStack.name}."
            )
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = torch.nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = torch.nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = torch.nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )

        self.fc_neigh = torch.nn.Linear(self._in_src_feats, out_feats, bias=False)

        # TODO: replace lower code with upper code -- more up-to-date with DGL 1.x
        if aggregator_type != "gcn":
            self.fc_self = torch.nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = torch.nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            torch.nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            torch.nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes) -> Dict[str, torch.Tensor]:
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]

            msg_fn = _fn.copy_u("h", "m")
            # msg_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = _fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                graph.update_all(msg_fn, _fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)

            elif self._aggre_type == "gcn":
                if isinstance(feat, tuple):  # heterogeneous
                    assert feat[0].shape == feat[1].shape
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][: graph.num_dst_nodes()]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, _fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                    degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)

            elif self._aggre_type == "pool":
                graph.srcdata["h"] = torch.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, _fn.max("m", "neigh"))
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])

            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])

            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(self._aggre_type)
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)

            return rst


class SAGEConvStack(BaseGCNStack[Union[SAGEConv, "dgl.nn.pytorch.SAGEConv"]]):
    """
    GraphSAGE graph convolutional neural network for atom embeddings.

    `GraphSAGE <https://snap.stanford.edu/graphsage/>`_ GCNs learn a function
    that iteratively improves a node embedding by mixing in aggregated feature
    vectors of progressively more distant neighborhoods. GraphSAGE is inductive,
    scales to large graphs, and makes good use of feature-rich node embeddings.

    Layers in this network use the DGL :py:class:`SAGEConv
    <dgl.nn.pytorch.conv.SAGEConv>` class.

    See Also
    --------
    dgl.nn.pytorch.conv.SAGEConv
    """

    name = "SAGEConv"
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
    ) -> Union[SAGEConv, "dgl.nn.pytorch.SAGEConv"]:
        try:
            return cls._create_gcn_layer_dgl(
                n_input_features=n_input_features,
                n_output_features=n_output_features,
                aggregator_type=aggregator_type,
                dropout=dropout,
                activation_function=activation_function,
                **kwargs,
            )
        except MissingOptionalDependencyError:
            return cls._create_gcn_layer_nagl(
                n_input_features=n_input_features,
                n_output_features=n_output_features,
                aggregator_type=aggregator_type,
                dropout=dropout,
                activation_function=activation_function,
                **kwargs,
            )

    @classmethod
    def _create_gcn_layer_nagl(
        cls,
        n_input_features: int,
        n_output_features: int,
        aggregator_type: str,
        dropout: float,
        activation_function: ActivationFunction,
        **kwargs,
    ) -> "SAGEConv":
        return SAGEConv(
            in_feats=n_input_features,
            out_feats=n_output_features,
            activation=activation_function,
            feat_drop=dropout,
            aggregator_type=aggregator_type,
        )

    @classmethod
    @requires_package("dgl")
    def _create_gcn_layer_dgl(
        cls,
        n_input_features: int,
        n_output_features: int,
        aggregator_type: str,
        dropout: float,
        activation_function: ActivationFunction,
        **kwargs,
    ) -> "dgl.nn.pytorch.SAGEConv":
        import dgl

        return dgl.nn.pytorch.SAGEConv(
            in_feats=n_input_features,
            out_feats=n_output_features,
            activation=activation_function,
            feat_drop=dropout,
            aggregator_type=aggregator_type,
        )

    def _as_nagl(self, copy_weights: bool = False):
        if self._is_dgl:
            new_obj = type(self)()
            new_obj.hidden_feature_sizes = self.hidden_feature_sizes
            for layer in self:
                new_layer = self._create_gcn_layer_nagl(
                    n_input_features=layer._in_src_feats,
                    n_output_features=layer._out_feats,
                    aggregator_type=layer._aggre_type,
                    dropout=layer.feat_drop.p,
                    activation_function=layer.activation,
                )
                if copy_weights:
                    new_layer.load_state_dict(layer.state_dict())
                new_obj.append(new_layer)
            return copy.deepcopy(new_obj)
        return copy.deepcopy(self)
