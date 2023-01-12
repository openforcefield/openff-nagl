import copy
from collections import defaultdict, UserDict
from typing import List, Tuple, Optional, Dict

from openff.toolkit import Molecule
from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.features._featurizers import AtomFeaturizer, BondFeaturizer

import networkx as nx
import torch
from ._batch import FrameDict

from openff.nagl.molecule._utils import FORWARD, REVERSE, FEATURE


def openff_molecule_to_base_nx_graph(
    molecule: Molecule,
    forward: str = FORWARD,
    reverse: str = REVERSE,
):
    graph = molecule.to_networkx()
    return graph


class NXMolGraph:
    def __init__(self, graph: nx.Graph):
        self._graph = nx.freeze(graph)
        self.graph["node_data"] = FrameDict(self.graph.get("node_data", {}))
        self.graph["edge_data"] = FrameDict(self.graph.get("edge_data", {}))
        self.graph["graph_data"] = FrameDict(self.graph.get("graph_data", {}))
        self.__post_init__()
        
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

    def in_degrees(self):
        return self._degrees

    def _get_degrees(self):
        return torch.tensor([
            degree
            for _, degree
            in self.graph.degree()
        ], dtype=torch.int32)
        

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

    def _bond_indices(self):
        u, v = map(list, zip(*self.graph.edges()))
        U = torch.tensor(u, dtype=torch.int32)
        V = torch.tensor(v, dtype=torch.int32)
        return U, V

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

    def dstnodes(self):
        return self._bond_indices()[1]

    @property
    def srcdata(self):
        return self.ndata
    
    @property
    def dstdata(self):
        return self.ndata

class NXMolHomoGraph(NXMolGraph):
    def _bond_indices(self):
        u, v = super()._bond_indices()
        U = torch.cat([u, v], dim=0)
        V = torch.cat([v, u], dim=0)
        return U, V


class NXMolHeteroGraph(NXMolGraph):
    def __init__(self, graph: nx.Graph):
        super().__init__(graph)
        self.graph["edge_data"] = defaultdict(FrameDict)
        self.__post_init__()
    
    @property
    def srcdata(self):
        raise NotImplementedError

    @property
    def dstdata(self):
        raise NotImplementedError
    

    @classmethod
    def from_openff(
        cls,
        molecule: Molecule,
        atom_features: Tuple[AtomFeature, ...] = tuple(),
        bond_features: Tuple[BondFeature, ...] = tuple(),
        enumerate_resonance_forms: bool = False,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
    ):
        from openff.nagl.molecule._utils import _get_openff_molecule_information
        from openff.nagl.utils.resonance import ResonanceEnumerator

        offmols = [molecule]
        if enumerate_resonance_forms:
            enumerator = ResonanceEnumerator(molecule)
            offmols = enumerator.enumerate_resonance_forms(
                lowest_energy_only=lowest_energy_only,
                max_path_length=max_path_length,
                include_all_transfer_pathways=include_all_transfer_pathways,
                as_dicts=False,
            )

        
        nx_graphs = [
            openff_molecule_to_base_nx_graph(mol)
             for mol in offmols
        ]
        joined_graph = nx.disjoint_union_all(nx_graphs)
        molecule_graph = cls(joined_graph)

        if len(atom_features):
            atom_featurizer = AtomFeaturizer(atom_features)

            atom_features = [
                atom_featurizer.featurize(mol)
                for mol in offmols
            ]
            overall_features = torch.cat(atom_features, dim=0)
            molecule_graph.ndata[FEATURE] = overall_features
        
        molecule_info = _get_openff_molecule_information(molecule)
        for key, value in molecule_info.items():
            tiled_value = torch.tensor(list(value) * len(offmols))
            molecule_graph.ndata[key] = tiled_value
       
       # add bond features
        single_bond_orders = [bond.bond_order for bond in molecule.bonds]
        bond_orders = torch.tensor(
            single_bond_orders * len(offmols), dtype=torch.uint8
        )
        molecule_graph.edges[FORWARD].data["bond_order"] = bond_orders
        molecule_graph.edges[REVERSE].data["bond_order"] = bond_orders

        if len(bond_features):
            bond_featurizer = BondFeaturizer(bond_features)
            bond_feature_tensor = [
                bond_featurizer.featurize(mol)
                for mol in offmols
            ]
            overall_features = torch.cat(bond_feature_tensor, dim=0)
            molecule_graph.edges[FORWARD].data[FEATURE] = overall_features
            molecule_graph.edges[REVERSE].data[FEATURE] = overall_features

        return molecule_graph
    
    def to_homogeneous(self):
        graph = copy.deepcopy(self.graph)

        # only keep FEATURE information
        non_features = [x for x in self.ndata if x != FEATURE]
        for non_feature in non_features:
            del graph["node_data"][non_feature]

        all_edge_features = []
        for edge_features in self.edges.values():
            if FEATURE in edge_features:
                all_edge_features.append(edge_features[FEATURE])
        graph["edge_data"] = FrameDict()
        if len(all_edge_features):
            graph["edge_data"][FEATURE] = torch.cat(all_edge_features, dim=0)
        return NXMolHomoGraph(graph)