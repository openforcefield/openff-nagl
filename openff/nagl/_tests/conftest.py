"""
Global pytest fixtures
"""


import numpy as np
import pytest

from openff.nagl.molecule._dgl.molecule import DGLMolecule
from openff.nagl.molecule._dgl.batch import DGLMoleculeBatch
from openff.nagl.features.atoms import AtomConnectivity, AtomicElement
from openff.nagl.features.bonds import BondIsInRing


@pytest.fixture()
def openff_methyl_methanoate():
    from openff.toolkit.topology.molecule import Molecule

    mapped_smiles = "[H:5][C:1](=[O:2])[O:3][C:4]([H:6])([H:7])[H:8]"
    return Molecule.from_mapped_smiles(mapped_smiles)


@pytest.fixture()
def openff_methane_uncharged():
    from openff.toolkit.topology.molecule import Molecule, unit

    molecule = Molecule.from_smiles("C")
    molecule.add_conformer(
        np.array(
            [
                [-0.0000658, -0.0000061, 0.0000215],
                [-0.0566733, 1.0873573, -0.0859463],
                [0.6194599, -0.3971111, -0.8071615],
                [-1.0042799, -0.4236047, -0.0695677],
                [0.4415590, -0.2666354, 0.9626540],
            ]
        )
        * unit.angstrom
    )

    return molecule


@pytest.fixture()
def openff_methane_charges():
    return np.arange(5, dtype=float) / 10


@pytest.fixture()
def openff_methane_charged(openff_methane_uncharged, openff_methane_charges):
    from openff.toolkit.topology.molecule import unit

    charges = openff_methane_charges * unit.elementary_charge
    openff_methane_uncharged.partial_charges = charges

    return openff_methane_uncharged


@pytest.fixture()
def dgl_methane(openff_methane_uncharged):
    pytest.importorskip("dgl")
    return DGLMolecule.from_openff(
        openff_methane_uncharged,
        atom_features=[AtomConnectivity()],
        bond_features=[BondIsInRing()],
    )

@pytest.fixture()
def nx_methane(openff_methane_uncharged):
    from openff.nagl.molecule._graph.molecule import GraphMolecule

    return GraphMolecule.from_openff(
        openff_methane_uncharged,
        atom_features=[AtomConnectivity()],
        bond_features=[BondIsInRing()],
    )


@pytest.fixture()
def openff_carboxylate():
    from openff.toolkit.topology.molecule import Molecule

    return Molecule.from_mapped_smiles("[H:1][C:2](=[O:3])[O-:4]")


@pytest.fixture()
def openff_ccnco():
    from openff.toolkit.topology.molecule import Molecule

    return Molecule.from_mapped_smiles("[H:6][C:1]([H:7])([H:8])[C:2]([H:9])([H:10])[N:3]([H:11])[C:4]([H:12])([H:13])[O:5][H:14]")


@pytest.fixture()
def openff_cnc():
    from openff.toolkit.topology.molecule import Molecule

    return Molecule.from_mapped_smiles("[H:4][C:1]([H:5])([H:6])[N:2]([H:7])[C:3]([H:8])([H:9])[H:10]")


@pytest.fixture()
def example_atom_features():
    return [AtomicElement(), AtomConnectivity()]

@pytest.fixture()
def example_bond_features():
    return [BondIsInRing()]

@pytest.fixture()
def dgl_carboxylate(openff_carboxylate):
    pytest.importorskip("dgl")
    return DGLMolecule.from_openff(
        openff_carboxylate,
        atom_features=[AtomConnectivity()],
        bond_features=[BondIsInRing()],
    )

@pytest.fixture()
def dgl_ccnco(openff_ccnco):
    pytest.importorskip("dgl")
    return DGLMolecule.from_openff(
        openff_ccnco,
        atom_features=[AtomConnectivity()],
        bond_features=[BondIsInRing()],
    )


@pytest.fixture()
def dgl_batch(dgl_methane, dgl_carboxylate, dgl_ccnco):
    pytest.importorskip("dgl")
    return DGLMoleculeBatch.from_dgl_molecules(
        [dgl_methane, dgl_carboxylate, dgl_ccnco]
    )