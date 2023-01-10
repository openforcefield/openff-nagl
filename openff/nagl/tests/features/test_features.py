import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from openff.toolkit.topology.molecule import Molecule as OFFMolecule
from openff.toolkit.topology.molecule import unit as offunit

from openff.nagl.features.atoms import (
    AtomAverageFormalCharge,
    AtomConnectivity,
    AtomFormalCharge,
    AtomHybridization,
    AtomicElement,
    AtomInRingOfSize,
    AtomIsAromatic,
    AtomIsInRing,
)
from openff.nagl.features.bonds import (
    BondInRingOfSize,
    BondIsAromatic,
    BondIsInRing,
    BondOrder,
    WibergBondOrder,
)
from openff.nagl.utils.types import HybridizationType


@pytest.fixture()
def openff_benzene():
    return OFFMolecule.from_smiles("c1ccccc1")


@pytest.mark.parametrize(
    "smiles, connectivity",
    [
        ("C#C", [[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
        ("[H]Cl", [[1, 0, 0, 0], [1, 0, 0, 0]]),
    ],
)
def test_atom_connectivity(smiles, connectivity):
    feature = AtomConnectivity()
    assert len(feature) == 4

    offmol = OFFMolecule.from_smiles(smiles)
    assert_equal(feature(offmol).numpy(), connectivity)


@pytest.mark.parametrize(
    "smiles, hybridization",
    [
        # sp
        ("C#C", [[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
        # sp2
        ("B", [[0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
        # sp3
        ("C", [[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
    ],
)
def test_atom_hybridization(smiles, hybridization):
    feature = AtomHybridization(categories=["other", "sp", "sp2", "sp3"])
    assert len(feature) == 4

    offmol = OFFMolecule.from_smiles(smiles)
    assert_equal(feature(offmol).numpy(), hybridization)


@pytest.mark.parametrize(
    "smiles, charge",
    [
        ("[H+]", [[0, 0, 1]]),
        ("[NH2-]", [[0, 1, 0], [1, 0, 0], [1, 0, 0]]),
    ],
)
def test_atom_formal_charge(smiles, charge):
    feature = AtomFormalCharge(categories=[0, -1, 1])
    assert len(feature) == 3

    offmol = OFFMolecule.from_smiles(smiles)
    assert_equal(feature(offmol).numpy(), charge)


def test_atom_average_formal_charge():
    smiles = "[C:1](=[O:4])([O-:5])[C:2]([H:8])([H:9])[C:3](=[O:6])([O-:7])"
    offmol = OFFMolecule.from_mapped_smiles(smiles)
    feature = AtomAverageFormalCharge()
    charges = feature(offmol).numpy()

    expected = np.array([[0, 0, 0, -0.5, -0.5, -0.5, -0.5, 0, 0]])
    assert_allclose(charges.T, expected)


@pytest.mark.parametrize(
    "feature_class", [AtomIsAromatic, AtomIsInRing, BondIsAromatic, BondIsInRing]
)
def test_is_aromatic_and_is_in_ring(openff_benzene, feature_class):
    feature = feature_class()
    assert len(feature) == 1

    encoding = feature(openff_benzene).numpy()
    assert encoding.shape == (12, 1)

    assert_equal(encoding[:6], 1)
    assert_equal(encoding[6:], 0)


def test_bond_order():
    feature = BondOrder(categories=[2, 1])
    assert len(feature) == 2

    offmol = OFFMolecule.from_smiles("C=O")
    encoding = feature(offmol).numpy()
    assert encoding.shape == (3, 2)

    bond_orders = [[1, 0], [0, 1], [0, 1]]
    assert_equal(encoding, bond_orders)


def test_wiberg_bond_order(openff_methane_uncharged):
    for i, bond in enumerate(openff_methane_uncharged.bonds):
        bond.fractional_bond_order = float(i)

    feature = WibergBondOrder()
    assert len(feature) == 1

    encoding = feature(openff_methane_uncharged).numpy()
    assert encoding.shape == (4, 1)

    assert_allclose(encoding, [[0], [1], [2], [3]])
