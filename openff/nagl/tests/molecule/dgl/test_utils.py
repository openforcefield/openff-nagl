import pytest
import torch
from openff.toolkit.topology.molecule import Molecule
from torch.testing import assert_close

from openff.nagl.molecule._dgl.utils import (
    dgl_heterograph_to_homograph,
    openff_molecule_to_base_dgl_graph,
    openff_molecule_to_dgl_graph,
)
from openff.nagl.features.atoms import AtomConnectivity
from openff.nagl.features.bonds import BondIsInRing

@pytest.fixture()
def methane_dgl_heterograph():
    offmol = Molecule.from_smiles("C")
    graph = openff_molecule_to_base_dgl_graph(offmol)
    return graph


def test_openff_molecule_to_base_dgl_graph(methane_dgl_heterograph):
    assert methane_dgl_heterograph.number_of_nodes() == 5
    assert methane_dgl_heterograph.number_of_edges() == 8


@pytest.mark.parametrize(
    "atom_features, bond_features",
    [
        ([], []),
        ([AtomConnectivity()], []),
        ([], [BondIsInRing()]),
        ([AtomConnectivity()], [BondIsInRing()]),
    ],
)
def test_openff_molecule_to_dgl_graph(
    openff_methane_uncharged, atom_features, bond_features
):
    graph = openff_molecule_to_dgl_graph(
        openff_methane_uncharged, atom_features, bond_features
    )
    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 8

    if len(atom_features):
        atom_tensor = graph.ndata["feat"]
        assert atom_tensor.shape == ((5, 4))
        expected = torch.Tensor(
            [[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
        )
        assert_close(atom_tensor.type(torch.float32), expected)

    for direction in ("forward", "reverse"):
        if len(bond_features):
            bond_tensor = graph.edges[direction].data["feat"]
            assert_close(bond_tensor, torch.zeros((4, 1), dtype=bool))

        bond_orders = graph.edges[direction].data["bond_order"]
        assert_close(bond_orders, torch.ones((4,), dtype=torch.uint8))


def test_dgl_heterograph_to_homograph(methane_dgl_heterograph):
    assert methane_dgl_heterograph.number_of_nodes() == 5
    assert methane_dgl_heterograph.number_of_edges() == 8

    homograph = dgl_heterograph_to_homograph(methane_dgl_heterograph)
    assert homograph.is_homogeneous
    assert homograph.number_of_nodes() == 5
    assert homograph.number_of_edges() == 8  # 4 forward + 4 reverse
    indices_a, indices_b = homograph.edges()

    assert torch.allclose(indices_a[:4], indices_b[4:])
    assert torch.allclose(indices_b[4:], indices_a[:4])
