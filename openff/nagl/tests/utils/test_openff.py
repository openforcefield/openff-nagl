import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from openff.toolkit.topology import Molecule
from openff.nagl.toolkits import NAGLRDKitToolkitWrapper
from openff.toolkit import RDKitToolkitWrapper
from openff.toolkit.utils.toolkit_registry import toolkit_registry_manager, ToolkitRegistry
from openff.toolkit.utils.toolkits import RDKIT_AVAILABLE
from openff.units import unit

from openff.nagl.toolkits.openff import (
    get_best_rmsd,
    get_openff_molecule_bond_indices,
    is_conformer_identical,
    map_indexed_smiles,
    normalize_molecule,
    smiles_to_inchi_key,
    calculate_circular_fingerprint_similarity,
    capture_toolkit_warnings,
    molecule_from_networkx,
    _molecule_from_dict,
    _molecule_to_dict,
    split_up_molecule,
)
from openff.nagl.utils._utils import transform_coordinates


def test_get_openff_molecule_bond_indices(openff_methane_charged):
    bond_indices = get_openff_molecule_bond_indices(openff_methane_charged)
    assert bond_indices == [(0, 1), (0, 2), (0, 3), (0, 4)]


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("Cl", "VEXZGXHMUGYJMC-UHFFFAOYNA-N"),
        ("[H]Cl", "VEXZGXHMUGYJMC-UHFFFAOYNA-N"),
        ("[Cl:2][H:1]", "VEXZGXHMUGYJMC-UHFFFAOYNA-N"),
        ("C", "VNWKTOKETHGBQD-UHFFFAOYNA-N"),
        ("[CH4]", "VNWKTOKETHGBQD-UHFFFAOYNA-N"),
    ],
)
def test_smiles_to_inchi_key(smiles, expected):
    assert smiles_to_inchi_key(smiles) == expected


@pytest.mark.parametrize(
    "expected_smiles, given_smiles",
    [
        ("CS(=O)(=O)C", "C[S+2]([O-])([O-])C"),
    ],
)
def test_normalize_molecule(expected_smiles, given_smiles):
    from openff.toolkit.topology.molecule import Molecule
    expected_molecule = Molecule.from_smiles(expected_smiles)

    molecule = Molecule.from_smiles(given_smiles)
    assert not Molecule.are_isomorphic(molecule, expected_molecule)[0]

    output_molecule = normalize_molecule(molecule)
    assert Molecule.are_isomorphic(output_molecule, expected_molecule)[0]


@pytest.mark.parametrize(
    "smiles_a,smiles_b,expected",
    [
        ("[Cl:1][H:2]", "[Cl:2][H:1]", {0: 1, 1: 0}),
        ("[Cl:2][H:1]", "[Cl:1][H:2]", {0: 1, 1: 0}),
    ],
)
def test_map_indexed_smiles(smiles_a, smiles_b, expected):
    assert map_indexed_smiles(smiles_a, smiles_b) == expected


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccc(cc1)c2ccccc2",
        "c1ccccc1",
        "O=C(N)N",
        "CCC",
    ],
)
def test_is_conformer_identical_generated(smiles):
    from openff.toolkit.topology.molecule import Molecule

    offmol = Molecule.from_smiles(smiles)
    offmol.generate_conformers(n_conformers=1)
    ordered_conf = offmol.conformers[0].m_as(unit.angstrom)
    # ordered_conf = get_coordinates_in_angstrom(offmol.conformers[0])

    # Create a permuted version of the conformer,
    # permuting only topology symmetric atoms.
    indexed_smiles = offmol.to_smiles(isomeric=False, mapped=True)
    matches = offmol.chemical_environment_matches(indexed_smiles)
    permuted_indices = max(matches)
    ordered_indices = tuple(range(len(permuted_indices)))
    assert permuted_indices != ordered_indices, "No permutation found"

    transformed_conf = transform_coordinates(
        ordered_conf.copy(),
        scale=1,
        translate=np.random.random(),
        rotate=np.random.random(),
    )
    permuted_conf = transformed_conf[permuted_indices, :]
    assert is_conformer_identical(offmol, ordered_conf, ordered_conf)
    assert is_conformer_identical(offmol, ordered_conf, transformed_conf)
    assert is_conformer_identical(offmol, ordered_conf, permuted_conf)
    assert not is_conformer_identical(offmol, ordered_conf, permuted_conf * 2.0)


def test_is_conformer_identical_linear():
    from openff.toolkit.topology.molecule import Molecule

    offmol = Molecule.from_smiles("CCC")
    c_coords = np.array(
        [
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
        ],
        dtype=float,
    )
    ordered_conf = np.vstack([c_coords, np.random.random((8, 3))])
    permuted_indices = [2, 1, 0, 10, 9, 8, 7, 6, 5, 4, 3]
    permuted_conf = transform_coordinates(
        ordered_conf.copy(),
        scale=1,
        translate=np.random.random(),
        rotate=np.random.random(),
    )[permuted_indices]

    assert is_conformer_identical(offmol, ordered_conf, permuted_conf)
    assert not is_conformer_identical(offmol, ordered_conf, permuted_conf * 2.0)


def test_not_is_conformer_identical():
    from openff.toolkit.topology.molecule import Molecule

    smiles = "[C:1]([H:4])([H:5])([H:6])[C:2]([Cl:7])=[O:3]"
    offmol = Molecule.from_mapped_smiles(smiles)
    offmol.generate_conformers(n_conformers=1)

    conformer = offmol.conformers[0].m_as(unit.angstrom)

    # Swap and perturb the hydrogen positions.
    hydrogen_coordinates = conformer[3, :]

    perturbed_conformer = conformer.copy()
    perturbed_conformer[3, :] = perturbed_conformer[4, :]
    perturbed_conformer[4, :] = hydrogen_coordinates + 0.1

    assert not is_conformer_identical(offmol, conformer, perturbed_conformer)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "smiles1, smiles2, radius, similarity",
    [
        ("C", "C", 3, 1.0),
        ("C", "N", 3, 0.33333333333333333),
    ],
)
def test_calculate_circular_fingerprint_similarity(
    smiles1, smiles2, radius, similarity
):
    from openff.toolkit.topology.molecule import Molecule

    mol1 = Molecule.from_smiles(smiles1)
    mol2 = Molecule.from_smiles(smiles2)

    dice = calculate_circular_fingerprint_similarity(mol1, mol2, radius=radius)
    assert_allclose(dice, similarity)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_get_best_rmsd():
    from rdkit.Chem import rdMolAlign
    from openff.toolkit.topology.molecule import Molecule

    offmol = Molecule.from_smiles("CCC")
    offmol._conformers = [
        np.random.random((11, 3)) * unit.angstrom,
        np.random.random((11, 3)) * unit.angstrom,
    ]

    rdmol = offmol.to_rdkit()
    assert rdmol.GetNumConformers() == 2

    reference_rmsd = rdMolAlign.GetBestRMS(rdmol, rdmol, 0, 1)
    rmsd = get_best_rmsd(
        offmol,
        offmol.conformers[0].m_as(unit.angstrom),
        offmol.conformers[1].m_as(unit.angstrom),
    )
    assert_allclose(rmsd, reference_rmsd)


def test_capture_toolkit_warnings(caplog):
    from openff.toolkit.topology.molecule import Molecule

    caplog.clear()
    smiles = "ClC=CCl"
    stereo_warning = "Warning (not error because allow_undefined_stereo=True)"

    Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    # as of toolkit v0.14.4 this warning is no longer raised
    # assert len(caplog.records) == 1
    # assert stereo_warning in caplog.records[0].message

    caplog.clear()
    with capture_toolkit_warnings():
        Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    assert len(caplog.records) == 0

    # as of toolkit v0.14.4 this warning is no longer raised
    # check that logging goes back to normal outside context manager
    # Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    # assert len(caplog.records) == 1
    # assert stereo_warning in caplog.records[0].message

    # check we haven't messed with warnings
    with warnings.catch_warnings(record=True) as records:
        warnings.warn("test")
        assert len(records)

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_openff_toolkit_registry(openff_methane_uncharged):

    rdkit_registry = ToolkitRegistry([NAGLRDKitToolkitWrapper()])
    with toolkit_registry_manager(rdkit_registry):
        normalize_molecule(openff_methane_uncharged)


def test_molecule_from_networkx(openff_methane_uncharged):
    graph = openff_methane_uncharged.to_networkx()
    molecule = molecule_from_networkx(graph)
    assert len(molecule.atoms) == 5
    
    atomic_numbers = [atom.atomic_number for atom in molecule.atoms]
    assert atomic_numbers == [6, 1, 1, 1, 1]
    is_aromatic = [atom.is_aromatic for atom in molecule.atoms]
    assert is_aromatic == [False, False, False, False, False]
    formal_charges = [atom.formal_charge for atom in molecule.atoms]
    assert formal_charges == [0, 0, 0, 0, 0]
    bond_orders = [bond.bond_order for bond in molecule.bonds]
    assert bond_orders == [1, 1, 1, 1]

    assert molecule.is_isomorphic_with(openff_methane_uncharged)


def test_molecule_to_dict(openff_methane_uncharged):
    graph = _molecule_to_dict(openff_methane_uncharged)
    atoms = graph["atoms"]
    bonds = graph["bonds"]
    assert len(atoms) == 5
    assert len(bonds) == 4

    c = {
        "atomic_number": 6,
        "is_aromatic": False,
        "formal_charge": 0,
        "stereochemistry": None
    }
    h = {
        "atomic_number": 1,
        "is_aromatic": False,
        "formal_charge": 0,
        "stereochemistry": None
    }
    assert atoms[0] == c
    assert atoms[1] == h
    assert atoms[2] == h
    assert atoms[3] == h
    assert atoms[4] == h

    ch_bond = {
        "bond_order": 1,
        "is_aromatic": False,
        "stereochemistry": None,
    }

    assert bonds[(0, 1)] == ch_bond
    assert bonds[(0, 2)] == ch_bond
    assert bonds[(0, 3)] == ch_bond
    assert bonds[(0, 4)] == ch_bond


def test_molecule_from_dict(openff_methane_uncharged):
    graph = _molecule_to_dict(openff_methane_uncharged)
    molecule = _molecule_from_dict(graph)
    assert molecule.is_isomorphic_with(openff_methane_uncharged)

def test_split_up_molecule():
    # "N.c1ccccc1.C.CCN"
    mapped_smiles = (
        "[H:17][c:4]1[c:3]([c:2]([c:7]([c:6]([c:5]1[H:18])[H:19])[H:20])[H:15])[H:16]"
        ".[H:21][C:8]([H:22])([H:23])[H:24]"
        ".[H:25][C:9]([H:26])([H:27])[C:10]([H:28])([H:29])[N:11]([H:30])[H:31]"
        ".[H:12][N:1]([H:13])[H:14]"
    )
    molecule = Molecule.from_mapped_smiles(mapped_smiles)

    fragments, indices = split_up_molecule(molecule, return_indices=True)
    assert len(fragments) == 4

    # check order
    n = Molecule.from_smiles("N")
    benzene = Molecule.from_smiles("c1ccccc1")
    ethanamine = Molecule.from_smiles("CCN")
    methane = Molecule.from_smiles("C")

    assert fragments[0].is_isomorphic_with(n)
    assert fragments[1].is_isomorphic_with(benzene)
    assert fragments[2].is_isomorphic_with(methane)
    assert fragments[3].is_isomorphic_with(ethanamine)

    assert indices[0] == [0, 11, 12, 13]
    assert indices[1] == [1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19]
    assert indices[2] == [7, 20, 21, 22, 23]
    assert indices[3] == [8, 9, 10, 24, 25, 26, 27, 28, 29, 30]


