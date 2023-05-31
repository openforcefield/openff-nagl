import abc
import functools
from typing import ClassVar, Dict, Union, TYPE_CHECKING, Iterable

import torch.nn

from openff.nagl.molecule._dgl import DGLMolecule, DGLMoleculeBatch, DGLMoleculeOrBatch
from openff.nagl.nn._sequential import SequentialLayers

if TYPE_CHECKING:
    import dgl


class PoolingLayer(torch.nn.Module, abc.ABC):
    """A convenience class for pooling together node feature vectors produced by
    a graph convolutional layer.
    """

    n_feature_columns: ClassVar[int] = 0

    @abc.abstractmethod
    def forward(self, molecule: DGLMoleculeOrBatch) -> torch.Tensor:
        """Returns the pooled feature vector."""

    
    @abc.abstractmethod
    def get_nvalues_per_molecule(self, molecule: DGLMoleculeOrBatch) -> Iterable[int]:
        """Returns the number of values per molecule."""

class PoolAtomFeatures(PoolingLayer):
    """A convenience class for pooling the node feature vectors produced by
    a graph convolutional layer.

    This class simply returns the features "h" from the graphs node data.
    """

    n_feature_columns: ClassVar[int] = 1

    def forward(self, molecule: DGLMoleculeOrBatch) -> torch.Tensor:
        return molecule.graph.ndata[molecule._graph_feature_name]
    

    def get_nvalues_per_molecule(self, molecule: DGLMoleculeOrBatch) -> Iterable[int]:
        return molecule.n_atoms_per_molecule


class PoolBondFeatures(PoolingLayer):
    """A convenience class for pooling the node feature vectors produced by
    a graph convolutional layer into a set of symmetric bond (edge) features.
    """

    n_feature_columns: ClassVar[int] = 2

    def __init__(self, layers: SequentialLayers):
        super().__init__()
        self.layers = layers

    @staticmethod
    def _apply_edges(
        edges: "dgl.udf.EdgeBatch", feature_name: str = "h"
    ) -> Dict[str, torch.Tensor]:
        h_u = edges.src[feature_name]
        h_v = edges.dst[feature_name]
        return {feature_name: torch.cat([h_u, h_v], 1)}

    # def _directionwise_forward(
    #     self,
    #     molecule: DGLMoleculeOrBatch,
    #     edge_type: str = "forward",
    # ):
    #     graph = molecule.graph
    #     apply_edges = functools.partial(
    #         self._apply_edges,
    #         feature_name=molecule._graph_feature_name,
    #     )
    #     with graph.local_scope():
    #         graph.apply_edges(apply_edges, etype=edge_type)
    #         edges = graph.edges[edge_type].data[molecule._graph_feature_name]
    #     return self.layers(edges)

    def forward(self, molecule: DGLMoleculeOrBatch) -> torch.Tensor:
        graph = molecule.graph
        node = molecule._graph_feature_name
        apply_edges = functools.partial(
            self._apply_edges,
            feature_name=node,
        )

        with graph.local_scope():
            graph.apply_edges(apply_edges, etype=molecule._graph_forward_edge_type)
            h_forward = graph.edges[molecule._graph_forward_edge_type].data[node]

        with graph.local_scope():
            graph.apply_edges(apply_edges, etype=molecule._graph_backward_edge_type)
            h_reverse = graph.edges[molecule._graph_backward_edge_type].data[node]

        # h_forward = self._directionwise_forward(
        #     molecule,
        #     molecule._graph_forward_edge_type,
        # )
        # h_reverse = self._directionwise_forward(
        #     molecule,
        #     molecule._graph_backward_edge_type,
        # )
        return self.layers(h_forward) + self.layers(h_reverse)

    def get_nvalues_per_molecule(self, molecule: DGLMoleculeOrBatch) -> Iterable[int]:
        return molecule.n_bonds_per_molecule
    

def get_pooling_layer(layer: Union[str, PoolingLayer]) -> PoolingLayer:
    if isinstance(layer, PoolingLayer):
        return layer
    if isinstance(layer, str):
        if layer.lower() in {"atom", "atoms"}:
            return PoolAtomFeatures()
        if layer.lower() in {"bond", "bonds"}:
            return PoolBondFeatures()
    raise NotImplementedError(f"Unsupported pooling layer '{layer}'.")