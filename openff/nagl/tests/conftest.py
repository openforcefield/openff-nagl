"""
Global pytest fixtures
"""


import numpy as np
import pytest

from openff.nagl.dgl.molecule import DGLMolecule
from openff.nagl.features import AtomConnectivity, BondIsInRing


@pytest.fixture()
def openff_methane_uncharged():
    import openff.toolkit
    from openff.toolkit.topology.molecule import Molecule, unit

    print(openff.toolkit.__version__)
    print(openff.toolkit.__path__)

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
    return DGLMolecule.from_openff(
        openff_methane_uncharged,
        atom_features=[AtomConnectivity()],
        bond_features=[BondIsInRing()],
    )


@pytest.fixture()
def openff_carboxylate():
    from openff.toolkit.topology.molecule import Molecule

    return Molecule.from_mapped_smiles("[H:1][C:2](=[O:3])[O-:4]")
