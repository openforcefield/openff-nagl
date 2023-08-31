import numpy as np
import pytest
from numpy.testing import assert_allclose
from openff.toolkit.topology.molecule import Molecule
from openff.toolkit.utils.toolkits import RDKIT_AVAILABLE
from openff.units import unit

from openff.nagl.toolkits.openff import (
    calculate_circular_fingerprint_similarity,
    get_best_rmsd,
    get_openff_molecule_bond_indices,
    is_conformer_identical,
    map_indexed_smiles,
    normalize_molecule,
    smiles_to_inchi_key,
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
    mol1 = Molecule.from_smiles(smiles1)
    mol2 = Molecule.from_smiles(smiles2)

    dice = calculate_circular_fingerprint_similarity(mol1, mol2, radius=radius)
    assert_allclose(dice, similarity)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_get_best_rmsd():
    from rdkit.Chem import rdMolAlign

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
