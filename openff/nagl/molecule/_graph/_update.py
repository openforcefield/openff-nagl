"""
This module contains functions for updating the graph,
analogous to DGL's functionality.

Functions here are not intended to be called directly
and may be fragile.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np
import torch

from ._batch import EdgeBatch, FrameDict, NodeBatch

if TYPE_CHECKING:
    from ._graph import NXMolGraph

__all__ = ["message_passing"]


def apply_edge_function(
    nx_molecule: "NXMolGraph",
    edge_function: Callable[[EdgeBatch], Dict[str, torch.Tensor]],
):
    """
    Apply custom edge function to NXMolGraph.
    We assume that the function is applied to all
    edges of the molecule.

    Analogous to :func:``dgl.core.invoke_edge_udf``.

    #TODO: make this more general and analogous to DGL

    Parameters
    ----------
    nx_molecule : NXMolGraph
        The molecule graph.
    edge_function : callable
        This function is applied to a ``dgl.udf.EdgeBatch``
        or :class:``~openff.nagl.molecule._graph._batch.EdgeBatch``
        and returns a dictionary of tensors.

    """
    edge_batch = EdgeBatch._all_from_graph_molecule(nx_molecule)
    return edge_function(edge_batch)


def _order_edge_index_buckets(nx_molecule, degree, node_bucket):
    from .._graph._utils import as_numpy

    edge_index_buckets = nx_molecule.in_edges(node_bucket, form="eid")
    new_shape = (len(node_bucket), degree)
    edge_index_buckets = as_numpy(edge_index_buckets).reshape(new_shape)
    edge_index_buckets = np.sort(edge_index_buckets, axis=1)
    edge_index_buckets = torch.tensor(edge_index_buckets, dtype=torch.long)
    return edge_index_buckets


def apply_reduce_function(
    nx_molecule: "NXMolGraph",
    reduce_function: Callable[["NodeBatch"], Dict[str, torch.Tensor]],
    message_data: Dict[str, torch.Tensor],
    original_node_ids: Optional[torch.Tensor] = None,
) -> FrameDict:
    """
    Apply custom reduce function to NXMolGraph.
    Nodes are 'bucketed' by degree and the
    reduce function is applied to each bucket.

    Analogous to ``dgl.core.invoke_udf_reduce``

    Parameters
    ----------
    nx_molecule : NXMolGraph
        The molecule graph.
    reduce_function : callable
        This function is applied to a :class:``dgl.udf.NodeBatch``
        or :class:``~openff.nagl.molecule._graph._batch.NodeBatch``
        and returns a dictionary of tensors.
    message_data : dict
        This is a dictionary of tensors
        containing the message data for the entire graph.
    original_node_ids : torch.Tensor, optional
        The original node ids of the graph.
        If the passed graph is a subgraph of the original graph,
        this should be the node ids of the original graph.

    Returns
    -------
    FrameDict
        A dictionary of tensors containing the reduced data.
    """
    from .._graph._utils import _bucketing

    message_data = FrameDict(message_data)

    degrees = nx_molecule.in_degrees()
    unique_degrees, bucketor = _bucketing(degrees)

    # nodes = nx_molecule.dstnodes()
    nodes = np.array(nx_molecule.graph.nodes())
    nodes = torch.tensor(nodes, dtype=torch.long)
    if original_node_ids is None:
        original_node_ids = nodes

    node_buckets = bucketor(nodes)
    original_node_buckets = bucketor(nodes)

    bucket_nodes: List[torch.Tensor] = []
    bucket_results: List[torch.Tensor] = []
    for degree, node_bucket, original_node_bucket in zip(
        unique_degrees, node_buckets, original_node_buckets
    ):
        if degree == 0:  # skip 0-degree nodes
            continue

        bucket_nodes.append(node_bucket)
        node_data: dict = nx_molecule._node_data(node_bucket)

        # order incoming edges per node, by edge index
        edge_index_buckets = _order_edge_index_buckets(nx_molecule, degree, node_bucket)

        maildata = message_data.subframe(edge_index_buckets)
        node_batch = NodeBatch(
            nx_molecule, original_node_bucket, "_N", node_data, maildata
        )
        bucket_results.append(reduce_function(node_batch))

    # concatenate results into a new frame
    result = FrameDict()
    if len(bucket_results) != 0:
        merged_nodes = torch.cat(bucket_nodes, dim=0)
        for name in bucket_results[0]:
            tensor = torch.cat([bucket[name] for bucket in bucket_results], dim=0)
            result[name] = torch.empty_like(tensor)
            for i, j in zip(merged_nodes, tensor):
                result[name][i] = j
    return result


def message_passing(
    nx_molecule: "NXMolGraph",
    message_func,
    reduce_func,
    apply_func: Optional[callable] = None,
) -> FrameDict:
    """
    Perform message passing on a NXMolGraph.
    This is analogous to :func:``dgl.core.message_passing``.

    #TODO: incorporate apply_func

    Parameters
    ----------
    nx_molecule : NXMolGraph
        The molecule graph.
    message_func : callable
        This function is applied to a ``dgl.udf.EdgeBatch``
        or :class:``~openff.nagl.molecule._graph._batch.EdgeBatch``
        and returns a dictionary of tensors.
    reduce_func : callable
        This function is applied to a ``dgl.udf.NodeBatch``
        or :class:``~openff.nagl.molecule._graph._batch.NodeBatch``
        and returns a dictionary of tensors.
    apply_func : callable, optional
        This option is not currently supported
        and only provided for compatibility with DGL.

    Returns
    -------
    FrameDict
        A dictionary of tensors containing the final data.
    """
    message_data = apply_edge_function(nx_molecule, message_func)
    node_data = apply_reduce_function(nx_molecule, reduce_func, message_data)
    if apply_func is not None:
        raise NotImplementedError("Apply function not implemented")
    return node_data
