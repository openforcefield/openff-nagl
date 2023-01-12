from typing import TYPE_CHECKING, Dict, Optional, List

import torch
import numpy as np

from ._batch import EdgeBatch, NodeBatch, FrameDict

if TYPE_CHECKING:
    from ._graph import NXMolGraph


def apply_edge_function(
    nxmolecule: "NXMolGraph",
    edge_function
):
    edge_batch = EdgeBatch._all_from_networkx_molecule(nxmolecule)
    return edge_function(edge_batch)


def _order_edge_index_buckets(nxmolecule, degree, node_bucket):
    from .._graph._utils import as_numpy
    edge_index_buckets = nxmolecule.in_edges(node_bucket, form="eid")
    new_shape = (len(node_bucket), degree)
    edge_index_buckets = as_numpy(edge_index_buckets).reshape(new_shape)
    edge_index_buckets = np.sort(edge_index_buckets, axis=1)
    edge_index_buckets = torch.Tensor(edge_index_buckets)
    return edge_index_buckets

def apply_reduce_function(
    nxmolecule: "NXMolGraph",
    reduce_function: callable,
    message_data: Dict[str, torch.Tensor],
    original_node_ids: Optional[torch.Tensor] = None
) -> FrameDict:
    from .._graph._utils import _bucketing

    message_data = FrameDict(message_data)

    degrees = nxmolecule.in_degrees()
    unique_degrees, bucketor = _bucketing(degrees)

    nodes = nxmolecule.dstnodes()
    if original_node_ids is None:
        original_node_ids = nodes

    node_buckets = bucketor(nodes)
    original_node_buckets = bucketor(nodes)

    bucket_nodes: List[torch.Tensor] = []
    bucket_results: List[torch.Tensor] = []
    for degree, node_bucket, original_node_bucket in zip(
        unique_degrees,
        node_buckets,
        original_node_buckets
    ):
        if degree == 0:  # skip 0-degree nodes
            continue
            
        bucket_nodes.append(node_bucket)
        node_data: dict = nxmolecule._node_data(node_bucket)

        # order incoming edges per node, by edge index
        edge_index_buckets = _order_edge_index_buckets(
            nxmolecule, degree, node_bucket
        )

        bucket_message_data = message_data.subframe(edge_index_buckets)
        maildata = {}
        # reshape tensors to (n_nodes, degree, feature_size ...)
        for name, tensor in bucket_message_data.items():
            new_shape = (len(node_bucket), degree) + tensor.shape[1:]
            maildata[name] = tensor.reshape(new_shape)

        node_batch = NodeBatch(nxmolecule, original_node_bucket, "_N", node_data, maildata)
        bucket_results.append(reduce_function(node_batch))

    # concatenate results into a new frame
    result = FrameDict()
    if len(bucket_results) != 0:
    
        merged_nodes = torch.cat(bucket_nodes, dim=0)
        for name in bucket_results[0]:
            tensor = torch.cat(
                [
                    bucket[name]
                    for bucket in bucket_results
                ],
                dim=0
            )
            result[name] = tensor[merged_nodes]
    return result



def message_passing(
    nxmolecule: "NXMolGraph",
    message_func,
    reduce_func,
    apply_func: Optional[callable] = None
) -> FrameDict:

    message_data = apply_edge_function(nxmolecule, message_func)
    node_data = apply_reduce_function(nxmolecule, reduce_func, message_data)
    if apply_func is not None:
        raise NotImplementedError("Apply function not implemented")
    return node_data