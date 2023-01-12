from collections import defaultdict, UserDict
from typing import Optional, Dict
from typing import List

import torch
import numpy as np

from openff.utilities import requires_package
from .._graph._batch import EdgeBatch, NodeBatch 

class FrameDict(UserDict):

    def subframe(self, indices):
        return type(self)((key, value[indices]) for key, value in self.items())


def apply_edge_function(
    nxmolecule: "NetworkXMolecule",
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
    nxmolecule: "NetworkXMolecule",
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
    nxmolecule: "NetworkXMolecule",
    message_func,
    reduce_func,
    apply_func: Optional[callable] = None
) -> FrameDict:

    message_data = apply_edge_function(nxmolecule, message_func)
    node_data = apply_reduce_function(nxmolecule, reduce_func, message_data)
    if apply_func is not None:
        raise NotImplementedError("Apply function not implemented")
    return node_data


class NXMoleculeGraph:
    def __init__(self, graph):
        self._graph = graph
        self.graph["node_data"] = FrameDict()
        self.graph["edge_data"] = defaultdict(FrameDict)
        self.graph["graph_data"] = FrameDict()



class NXMolHeteroGraph(NXMoleculeGraph):
    ...



class MoleculeGraph:
    def __init__(self, graph):
        self._graph = graph
        self.graph["node_data"] = FrameDict()
        self.graph["edge_data"] = defaultdict(FrameDict)
        self.graph["graph_data"] = FrameDict()

        self._degrees = torch.tensor([
            degree
            for _, degree
            in self.graph.degree()
        ], dtype=torch.int32)

    @property
    def graph(self):
        return self._graph

    def _all_edges(self):
        u, v = map(list, zip(*self.graph.edges()))
        U = torch.tensor(u, dtype=torch.int32)
        V = torch.tensor(v, dtype=torch.int32)
        I = torch.arange(len(u), dtype=torch.int32)
        return U, V, I

    def in_edges(self, nodes, form="uv"):
        u, v, i = self._all_edges()
        mask = [x in nodes for x in v]
        U, V, I = u[mask], v[mask], i[mask]
        if form == "uv":
            return U, V
        elif form == "eid":
            return I
        elif form == "all":
            return U, V, I
        else:
            raise ValueError("Unknown form: {}".format(form))



class NetworkXMolecule(MoleculeGraph):



    

    @property
    def data(self):
        return self.graph.graph

    @property
    def ndata(self):
        return self.data["node_data"]

    @property
    def edges(self):
        return self.data["edge_data"]

    
    def in_degrees(self):
        return self._degrees


    
    def dstnodes(self):
        return self._bond_indices()[1]

    
    def update_all(self, message_func, reduce_func, apply_node_func=None, etype=None):
        updated_node_features = message_passing(
            self,
            message_func,
            reduce_func,
            apply_node_func
        )

    def apply_edges(self, func, edges=None, etype=None):
        ...

    def _bond_indices(self):
        u, v = map(list, zip(*self.graph.edges()))
        U = torch.tensor(u, dtype=torch.int32)
        V = torch.tensor(v, dtype=torch.int32)
        return U, V

    def _all_edges(self):
        # (tensor([0, 0, 0, 0, 1, 2, 3, 4], dtype=torch.int32),
        # tensor([1, 2, 3, 4, 0, 0, 0, 0], dtype=torch.int32),
        # tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32))

        u, v = self._bond_indices()
        U = torch.cat([u, v], dim=0)
        V = torch.cat([v, u], dim=0)
        I = torch.tensor(range(len(U)), dtype=torch.int32)
        return U, V, I



    def _node_data(self, nodes: List[int] = None):
        if nodes is None:
            nodes = list(self.graph.nodes())
        
        data = {
            k: v[nodes]
            for k, v in self.ndata
        }
        return data

    def _edge_data(self, edge_indices: List[int] = None):
        if edge_indices is None:
            edge_indices = list(range(self.graph.edges()))
        
        data = {
            k: v[edge_indices]
            for k, v in self.edata
        }
        return data

    @requires_package("dgl")
    def to_dgl_heterograph(self):
        import dgl

        u, v = self._bond_indices()


