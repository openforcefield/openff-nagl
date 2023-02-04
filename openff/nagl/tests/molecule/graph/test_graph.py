import pytest

from collections import defaultdict
import networkx as nx
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import torch

from openff.nagl.molecule._graph._batch import FrameDict
from openff.nagl.molecule._graph._graph import (
    NXMolGraph,
    NXMolHomoGraph,
    NXMolHeteroGraph,
)
from openff.nagl.features.atoms import AtomConnectivity
from openff.nagl.features.bonds import BondOrder


@pytest.fixture
def methyl_methanoate_hetero_graph(openff_methyl_methanoate):
    return NXMolHeteroGraph.from_openff(
        openff_methyl_methanoate,
        atom_features=[AtomConnectivity()],
        bond_features=[BondOrder()],
    )


@pytest.fixture
def methyl_methanoate_homo_graph(methyl_methanoate_hetero_graph):
    return methyl_methanoate_hetero_graph.to_homogeneous()


class TestNXMolGraph:
    def test_init_empty(self):
        graph = nx.Graph()
        molgraph = NXMolGraph(graph)
        assert not len(molgraph.edata)
        assert not len(molgraph.ndata)
        assert len(molgraph.data) == 3

        assert isinstance(molgraph.ndata, FrameDict)
        assert isinstance(molgraph.edata, FrameDict)
        assert molgraph.graph is not graph

    def test_init_with_existing_edge_data(self):
        graph = nx.Graph()
        graph.graph["edge_data"] = defaultdict(FrameDict)
        graph.graph["edge_data"]["key"] = "value"
        molgraph = NXMolGraph(graph)

        assert molgraph.edata == graph.graph["edge_data"]
        assert molgraph.edata is not graph.graph["edge_data"]
        assert molgraph.graph is not graph

    def test_local_scope(self):
        graph = nx.Graph()
        molgraph = NXMolGraph(graph)
        molgraph.edata["h"] = np.array([0, 1, 2])
        molgraph.edata["h2"] = np.array([3, 3, 3])
        with molgraph.local_scope():
            molgraph.edata["h"] += 1
            molgraph.edata["h2"] = np.array([4, 4, 4])
            molgraph.edata["h3"] = "test"

        # in-place change
        assert_equal(molgraph.edata["h"], np.array([1, 2, 3]))
        # no out-of-place change
        assert_equal(molgraph.edata["h2"], np.array([3, 3, 3]))
        # no new addition
        assert "h3" not in molgraph.edata


class TestNXMolHeteroGraph:
    def test_degree(self, methyl_methanoate_hetero_graph):
        degrees = methyl_methanoate_hetero_graph.in_degrees().detach().numpy()
        assert_equal(degrees, [3, 1, 2, 4, 1, 1, 1, 1])

    def test_bond_indices(self, methyl_methanoate_hetero_graph):
        u, v = methyl_methanoate_hetero_graph._bond_indices()

        U = u.detach().numpy()
        V = v.detach().numpy()

        expected_u = [
            0,
            0,
            0,
            2,
            3,
            3,
            3,
        ]
        expected_v = [
            1,
            2,
            4,
            3,
            5,
            6,
            7,
        ]
        assert_equal(U, expected_u)
        assert_equal(V, expected_v)

    def test_atom_data(self, methyl_methanoate_hetero_graph):
        atom_data = methyl_methanoate_hetero_graph.ndata["feat"].detach().numpy()
        expected_data = np.array(
            [
                [
                    0,
                    0,
                    1,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                ],
            ]
        )
        assert_allclose(atom_data, expected_data)

    def test_get_node_data(self, methyl_methanoate_hetero_graph):
        indices = torch.tensor([4, 0, 2])
        node_data = methyl_methanoate_hetero_graph._node_data(indices)
        expected_data = np.array(
            [
                [
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                ],
                [
                    0,
                    1,
                    0,
                    0,
                ],
            ]
        )
        assert_allclose(node_data["feat"].detach().numpy(), expected_data)

    def test_edge_data(self, methyl_methanoate_hetero_graph):
        edge_data = methyl_methanoate_hetero_graph.edata
        expected_data = np.array(
            [
                [
                    0,
                    1,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
            ]
        )
        assert len(edge_data) == 2
        forward = edge_data["forward"]["feat"].detach().numpy()
        reverse = edge_data["reverse"]["feat"].detach().numpy()
        assert_allclose(forward, expected_data)
        assert_allclose(reverse, expected_data)

    def test_num_nodes(self, methyl_methanoate_hetero_graph):
        assert methyl_methanoate_hetero_graph.number_of_nodes() == 8
        assert methyl_methanoate_hetero_graph.number_of_edges() == 7
        assert methyl_methanoate_hetero_graph.num_dst_nodes() == 8
        assert methyl_methanoate_hetero_graph.num_src_nodes() == 8

    def test_in_edges(self, methyl_methanoate_hetero_graph):
        nodes = [1, 3]
        u, v, i = methyl_methanoate_hetero_graph.in_edges(nodes, form="all")
        assert_equal(u.detach().numpy(), [0, 2])
        assert_equal(v.detach().numpy(), [1, 3])
        assert_equal(i.detach().numpy(), [0, 3])

        u_, v_ = methyl_methanoate_hetero_graph.in_edges(nodes, form="uv")
        assert torch.equal(u, u_)
        assert torch.equal(v, v_)

        i_ = methyl_methanoate_hetero_graph.in_edges(nodes, form="eid")
        assert torch.equal(i, i_)


class TestNXMolHomoGraph:
    def test_degree(self, methyl_methanoate_homo_graph):
        assert isinstance(methyl_methanoate_homo_graph, NXMolHomoGraph)
        degrees = methyl_methanoate_homo_graph.in_degrees().detach().numpy()
        assert_equal(degrees, [3, 1, 2, 4, 1, 1, 1, 1])

    def test_bond_indices(self, methyl_methanoate_homo_graph):
        u, v = methyl_methanoate_homo_graph._bond_indices()

        U = u.detach().numpy()
        V = v.detach().numpy()

        expected_u = [0, 0, 0, 2, 3, 3, 3, 1, 2, 4, 3, 5, 6, 7]
        expected_v = [1, 2, 4, 3, 5, 6, 7, 0, 0, 0, 2, 3, 3, 3]
        assert_equal(U, expected_u)
        assert_equal(V, expected_v)

    def test_atom_data(self, methyl_methanoate_hetero_graph):
        atom_data = methyl_methanoate_hetero_graph.ndata["feat"].detach().numpy()
        expected_data = np.array(
            [
                [
                    0,
                    0,
                    1,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                ],
            ]
        )
        assert_allclose(atom_data, expected_data)

    def test_get_node_data(self, methyl_methanoate_homo_graph):
        indices = torch.tensor([4, 0, 2])
        node_data = methyl_methanoate_homo_graph._node_data(indices)
        expected_data = np.array(
            [
                [
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                ],
                [
                    0,
                    1,
                    0,
                    0,
                ],
            ]
        )
        assert_allclose(node_data["feat"].detach().numpy(), expected_data)

    def test_edge_data(self, methyl_methanoate_homo_graph):
        edge_data = methyl_methanoate_homo_graph.edata["feat"].detach().numpy()
        expected_data = np.array(
            [
                [
                    0,
                    1,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    1,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
            ]
        )
        assert_allclose(edge_data, expected_data)

    def test_num_nodes(self, methyl_methanoate_homo_graph):
        assert methyl_methanoate_homo_graph.number_of_nodes() == 8
        assert methyl_methanoate_homo_graph.number_of_edges() == 14
        assert methyl_methanoate_homo_graph.num_dst_nodes() == 8
        assert methyl_methanoate_homo_graph.num_src_nodes() == 8

    def test_get_edge_data(self, methyl_methanoate_homo_graph):
        indices = torch.tensor([4, 0, 2])
        edge_data = methyl_methanoate_homo_graph._edge_data(indices)
        expected_data = np.array(
            [
                [
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    1,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                ],
            ]
        )
        assert_allclose(edge_data["feat"].detach().numpy(), expected_data)

    def test_in_edges(self, methyl_methanoate_homo_graph):
        nodes = [1, 3]
        u, v, i = methyl_methanoate_homo_graph.in_edges(nodes, form="all")
        assert_equal(u.detach().numpy(), [0, 2, 5, 6, 7])
        assert_equal(v.detach().numpy(), [1, 3, 3, 3, 3])
        assert_equal(i.detach().numpy(), [0, 3, 11, 12, 13])

        u_, v_ = methyl_methanoate_homo_graph.in_edges(nodes, form="uv")
        assert torch.equal(u, u_)
        assert torch.equal(v, v_)

        i_ = methyl_methanoate_homo_graph.in_edges(nodes, form="eid")
        assert torch.equal(i, i_)
