import pytest

import dgl
from numpy.testing import assert_allclose, assert_array_almost_equal

from gnn_charge_models.dgl.molecule import DGLMolecule
from gnn_charge_models.features import AtomConnectivity, BondIsInRing

class TestDGLBase:

    def test_graph_property(self, dgl_methane):
        assert isinstance(dgl_methane.graph, dgl.DGLHeteroGraph)

    def test_homograph_property(self, dgl_methane):
        assert isinstance(dgl_methane.graph, dgl.DGLHeteroGraph)
        assert dgl_methane.to_homogenous().is_homogeneous

    def test_atom_features_property(self, dgl_methane):
        assert dgl_methane.atom_features.shape == (5, 4)

    def test_to(self, dgl_methane):

        dgl_methane_to = dgl_methane.to("cpu")

        assert not dgl_methane_to is dgl_methane
        assert not dgl_methane_to.graph is dgl_methane.graph  # should be a copy.
        assert dgl_methane_to.n_atoms == 5
        assert dgl_methane_to.n_bonds == 4

class TestDGLMolecule:
    def test_n_properties(self):
        """Test that the number of atoms and bonds properties work correctly with
        multiple resonance structures"""

        dgl_molecule = DGLMolecule.from_smiles("[H]C(=O)[O-]", [], [])

        assert dgl_molecule.n_atoms == 4
        assert dgl_molecule.n_bonds == 3
        assert dgl_molecule.n_representations == 1

    def test_from_smiles(self):
        """Test that the DGLMolecule.from_smiles method works correctly"""

        dgl_molecule = DGLMolecule.from_smiles(
            "[H:1][C:2](=[O:3])[O-:4]",
            mapped=True,
            atom_features=[AtomConnectivity()],
            bond_features=[BondIsInRing()],

        )

        dgl_graph = dgl_molecule.graph

        node_features = dgl_molecule.atom_features
        assert node_features.shape == (4, 4)

        connectivity = [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ]

        assert_allclose(node_features, connectivity)

        forward_features = dgl_graph.edges["forward"].data["feat"].numpy()
        reverse_features = dgl_graph.edges["backward"].data["feat"].numpy()

        assert forward_features.shape == reverse_features.shape
        assert forward_features.shape == (3, 1)

        assert_allclose(forward_features, reverse_features)
        assert_array_almost_equal(forward_features, 0)
        