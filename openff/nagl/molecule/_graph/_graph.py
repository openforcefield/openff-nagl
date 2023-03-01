import contextlib
import copy
from collections import defaultdict
from typing import List, Tuple

from openff.toolkit import Molecule
from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.features._featurizers import AtomFeaturizer, BondFeaturizer

import networkx as nx
import torch
from ._batch import FrameDict

from openff.nagl.molecule._utils import FORWARD, REVERSE, FEATURE

__all__ = [
    "NXMolHeteroGraph",
    "NXMolHomoGraph",
]


def openff_molecule_to_base_nx_graph(
    molecule: Molecule,
    forward: str = FORWARD,
    reverse: str = REVERSE,
):
    graph = molecule.to_networkx()
    return graph


class NXMolGraph:
    """
    Base class for a NetworkX representation of a molecule graph.
    The API is intended to mimic DGL's :class:`dgl.heterograph.DGLGraph`
    as much as possible to facilitate sharing code.

    Graph data is stored as data on the graph dictionary itself;
    :attr:`NXMolGraph.ndata` is not stored on each individual node but
    as ``networkx.Graph.graph["node_data"]``. Similarly, :attr:`NXMolGraph.edata`
    is stored as ``networkx.Graph.graph["edge_data"]``.

    Where possible this class should not be directly instantiated
    by a user and **little effort has been made toward robustness**.
    Instead, this class is intended to be a drop-in replacement for
    :class:`dgl.heterograph.DGLGraph` during model inference
    in cases where `dgl` cannot be installed.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        The graph to wrap.
    """

    is_block = False

    def __init__(self, graph: nx.Graph, batch_size: int = 1):
        self.batch_size = batch_size
        self._graph = nx.freeze(copy.deepcopy(graph))
        self.data["node_data"] = FrameDict(self.data.get("node_data", {}))
        self.data["graph_data"] = FrameDict(self.data.get("graph_data", {}))
        self._set_edge_data()
        self.__post_init__()

    def _set_edge_data(self):
        """
        Separate method for initiating edge_data dictionary
        """
        edge_data = self.data.get("edge_data", {})
        if isinstance(edge_data, defaultdict):
            edge_data = defaultdict(FrameDict, edge_data)
        else:
            edge_data = FrameDict(edge_data)
        self.data["edge_data"] = edge_data

    @contextlib.contextmanager
    def local_scope(self):
        """
        Enter a local scope context for the graph. Any out-of-place changes
        will not be reflected in the graph. In-place changes will.
        """
        old_data = defaultdict(FrameDict)
        for k in ("node_data", "edge_data", "graph_data"):
            for k_, v_ in self.data[k].items():
                old_data[k][k_] = v_

        try:
            yield
        finally:
            for k, v in old_data.items():
                self.data[k] = v

    def __post_init__(self):
        self._degrees = self._get_degrees()

    @property
    def graph(self):
        return self._graph

    @property
    def data(self):
        return self.graph.graph

    @property
    def ndata(self):
        return self.data["node_data"]

    @property
    def edges(self):
        return self.data["edge_data"]

    @property
    def edata(self):
        return self.edges

    def in_degrees(self):
        return self._degrees

    def _get_degrees(self):
        return torch.tensor(
            [degree for _, degree in self.graph.degree()], dtype=torch.int32
        )

    def _all_edges(self):
        u, v = self._bond_indices()
        i = torch.arange(len(u), dtype=torch.long)
        return u, v, i

    def number_of_edges(self):
        return len(self._bond_indices()[0])

    def number_of_nodes(self):
        return len(self.graph.nodes())

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

    def _bond_indices(self):
        u, v = map(list, zip(*self.graph.edges()))
        U = torch.tensor(u, dtype=torch.long)
        V = torch.tensor(v, dtype=torch.long)
        return U, V

    def _node_data(self, nodes: List[int] = None):
        if nodes is None:
            nodes = torch.tensor(list(self.graph.nodes()))
        data = {k: v[nodes] for k, v in self.ndata.items()}
        return data

    def _edge_data(self, edge_indices: List[int] = None):
        if edge_indices is None:
            edge_indices = torch.tensor(list(range(self.graph.edges())))

        data = {k: v[edge_indices.long()].clone().detach() for k, v in self.edata.items()}
        return data

    def srcnodes(self):
        return torch.tensor(self.graph.nodes(), dtype=torch.int32)
        # return self._bond_indices()[0]

    def dstnodes(self):
        return torch.tensor(self.graph.nodes(), dtype=torch.int32)
        # return self._bond_indices()[1]

    @property
    def srcdata(self):
        return self.ndata

    @property
    def dstdata(self):
        return self.ndata

    def update_all(
        self,
        message_func,
        reduce_func,
        apply_node_func=None,
        etype=None,
    ):
        from ._update import message_passing

        results = message_passing(
            self,
            message_func,
            reduce_func,
            apply_node_func,
        )
        self.ndata.update(results)

    def apply_edges(self, func, edges="__ALL__", etype=None):
        raise NotImplementedError

    def num_dst_nodes(self):
        return len(self.dstnodes())

    def num_src_nodes(self):
        return len(self.srcnodes())


class NXMolHomoGraph(NXMolGraph):
    """
    NetworkX representation of a homogeneous molecule graph.
    The API is intended to mimic DGL's :class:`dgl.heterograph.DGLGraph`
    as much as possible to facilitate sharing code.

    Graph data is stored as data on the graph dictionary itself;
    :attr:`NXMolHomoGraph.ndata` is not stored on each individual node but
    as ``networkx.Graph.graph["node_data"]``. Similarly, :attr:`NXMolHomoGraph.edata`
    is stored as ``networkx.Graph.graph["edge_data"]``.

    Where possible this class should not be directly instantiated
    by a user and **little effort has been made toward robustness**.
    Instead, this class is intended to be a drop-in replacement for
    :class:`dgl.heterograph.DGLGraph` during model inference
    in cases where `dgl` cannot be installed.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        The graph to wrap.
    """

    def _bond_indices(self):
        u, v = super()._bond_indices()
        U = torch.cat([u, v], dim=0)
        V = torch.cat([v, u], dim=0)
        return U, V


class NXMolHeteroGraph(NXMolGraph):
    """
    Base class for a NetworkX representation of a molecule graph.
    The API is intended to mimic DGL's :class:`dgl.heterograph.DGLHeteroGraph`
    as much as possible to facilitate sharing code.

    Graph data is stored as data on the graph dictionary itself;
    :attr:`NXMolHeteroGraph.ndata` is not stored on each individual node but
    as ``networkx.Graph.graph["node_data"]``. Similarly, :attr:`NXMolHeteroGraph.edata`
    is stored as ``networkx.Graph.graph["edge_data"]``.

    Where possible this class should not be directly instantiated
    by a user and **little effort has been made toward robustness**.
    Instead, this class is intended to be a drop-in replacement for
    :class:`dgl.heterograph.DGLHeteroGraph` during model inference
    in cases where `dgl` cannot be installed.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        The graph to wrap.
    """

    def _set_edge_data(self):
        edge_data = self.data.get("edge_data", {})
        edge_data = defaultdict(FrameDict, edge_data)
        self.data["edge_data"] = edge_data

    @property
    def srcdata(self):
        raise NotImplementedError

    @property
    def dstdata(self):
        raise NotImplementedError

    @classmethod
    def _batch(cls, graphs: List["NXMolHeteroGraph"]) -> "NXMolHeteroGraph":
        from openff.nagl.molecule._graph._utils import _batch_nx_graphs

        if not graphs:
            return cls(nx.Graph())

        batched_graph = _batch_nx_graphs([g.graph for g in graphs])
        batched_graph = cls(batched_graph)
        return batched_graph

    @classmethod
    def from_openff(
        cls,
        molecule: Molecule,
        atom_features: Tuple[AtomFeature, ...] = tuple(),
        bond_features: Tuple[BondFeature, ...] = tuple(),
    ):
        from openff.nagl.molecule._utils import _get_openff_molecule_information

        nx_graph = openff_molecule_to_base_nx_graph(molecule)

        molecule_graph = cls(nx_graph)

        if len(atom_features):
            atom_featurizer = AtomFeaturizer(atom_features)
            atom_features = atom_featurizer.featurize(molecule)
            molecule_graph.ndata[FEATURE] = atom_features

        molecule_info = _get_openff_molecule_information(molecule)
        for key, value in molecule_info.items():
            molecule_graph.ndata[key] = value

        # add bond features
        bond_orders = torch.tensor(
            [bond.bond_order for bond in molecule.bonds], dtype=torch.uint8
        )
        molecule_graph.edges[FORWARD].data["bond_order"] = bond_orders
        molecule_graph.edges[REVERSE].data["bond_order"] = bond_orders

        if len(bond_features):
            bond_featurizer = BondFeaturizer(bond_features)
            bond_features = bond_featurizer.featurize(molecule)
            molecule_graph.edges[FORWARD].data[FEATURE] = bond_features
            molecule_graph.edges[REVERSE].data[FEATURE] = bond_features

        return molecule_graph

    def to_homogeneous(self):
        graph = copy.deepcopy(self.graph)

        # only keep FEATURE information
        non_features = [x for x in self.ndata if x != FEATURE]
        for non_feature in non_features:
            del graph.graph["node_data"][non_feature]

        all_edge_features = []
        for edge_features in self.edges.values():
            if FEATURE in edge_features:
                all_edge_features.append(edge_features[FEATURE])
        graph.graph["edge_data"] = FrameDict()
        if len(all_edge_features):
            graph.graph["edge_data"][FEATURE] = torch.cat(all_edge_features, dim=0)
        return NXMolHomoGraph(graph)
