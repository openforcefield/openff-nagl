from numpy.testing import assert_allclose, assert_array_almost_equal

from openff.nagl.features.atoms import AtomConnectivity
from openff.nagl.features.bonds import BondIsInRing
from openff.nagl.molecule._graph._graph import NXMolHeteroGraph, NXMolHomoGraph
from openff.nagl.molecule._graph.molecule import GraphMolecule


class TestNXMolecule:
    def test_homograph_property(self, nx_methane):
        assert isinstance(nx_methane.graph, NXMolHeteroGraph)
        assert isinstance(nx_methane.to_homogenous(), NXMolHomoGraph)

    def test_atom_features_property(self, nx_methane):
        assert nx_methane.atom_features.shape == (5, 4)

    def test_n_properties(self):
        """Test that the number of atoms and bonds properties work correctly with
        multiple resonance structures"""

        nx_molecule = GraphMolecule.from_smiles("[H]C(=O)[O-]", [], [])

        assert nx_molecule.n_atoms == 4
        assert nx_molecule.n_bonds == 3
        assert nx_molecule.n_representations == 1

    def test_from_smiles(self):
        """Test that the DGLMolecule.from_smiles method works correctly"""

        nx_molecule = GraphMolecule.from_smiles(
            "[H:1][C:2](=[O:3])[O-:4]",
            mapped=True,
            atom_features=[AtomConnectivity()],
            bond_features=[BondIsInRing()],
        )

        nx_graph = nx_molecule.graph

        node_features = nx_molecule.atom_features
        assert node_features.shape == (4, 4)

        connectivity = [[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0]]

        assert_allclose(node_features, connectivity)

        forward_features = nx_graph.edges["forward"].data["feat"].numpy()
        reverse_features = nx_graph.edges["reverse"].data["feat"].numpy()

        assert forward_features.shape == reverse_features.shape
        assert forward_features.shape == (3, 1)

        assert_allclose(forward_features, reverse_features)
        assert_array_almost_equal(forward_features, 0)
