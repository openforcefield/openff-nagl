import gzip
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from openff.toolkit.topology import Molecule
from openff.toolkit import Molecule, RDKitToolkitWrapper, AmberToolsToolkitWrapper, OpenEyeToolkitWrapper
from openff.nagl.toolkits import NAGLRDKitToolkitWrapper
from openff.toolkit.utils.toolkit_registry import toolkit_registry_manager, ToolkitRegistry
from openff.toolkit.utils.toolkits import RDKIT_AVAILABLE, OPENEYE_AVAILABLE
from openff.toolkit.utils.exceptions import MultipleComponentsInMoleculeWarning
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

from openff.nagl.tests.data.files import COFACTOR_SDF_GZ

def _load_rdkit_molecule_exactly(mapped_smiles: str):
    """
    Load a molecule from a mapped SMILES string using RDKit, without any normalization.
    """
    from rdkit import Chem

    # load into RDKit
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    rdmol = Chem.MolFromSmiles(mapped_smiles, params)
    Chem.MolToSmiles(rdmol)

    atom_indices = [atom.GetAtomMapNum() - 1 for atom in rdmol.GetAtoms()]
    ordering = [atom_indices.index(i) for i in range(rdmol.GetNumAtoms())]
    rdmol = Chem.RenumberAtoms(rdmol, ordering)
    Chem.SanitizeMol(rdmol, Chem.SANITIZE_SYMMRINGS)
    Chem.Kekulize(rdmol)

    molecule = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)

    # copy over formal charges and bonds again; from_rdkit sanitizes the rdmol
    for atom, rdatom in zip(molecule.atoms, rdmol.GetAtoms()):
        atom.formal_charge = rdatom.GetFormalCharge() * unit.elementary_charge
    for rdbond in rdmol.GetBonds():
        i, j = rdbond.GetBeginAtomIdx(), rdbond.GetEndAtomIdx()
        bond = molecule.get_bond_between(i, j)
        bond._bond_order = int(rdbond.GetBondTypeAsDouble())

    return molecule

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


NORMALIZATION_MOLECULE_TESTS = [
    (
        r"[H:6][C:1]([H:7])([H:8])[S+2:2]([C:5]([H:9])([H:10])[H:11])([O-:3])[O-:4]",
        r"[H:6][C:1]([H:7])([H:8])[S:2](=[O:3])(=[O:4])[C:5]([H:9])([H:10])[H:11]",

    ),
    (
        r"[H:22][c:7]1[c:6]([c:12]([n:10](=[O:11])[c:9]([n:8]1)[H:23])[C:13]([H:24])([H:25])[N:14](=[O:15])=[O:16])[C:5]([H:20])([H:21])[S+2:2]([C:1]([H:17])([H:18])[H:19])([O-:3])[O-:4]",
        r"[H:22][c:7]1[c:6]([c:12]([n+:10]([c:9]([n:8]1)[H:23])[O-:11])[C:13]([H:24])([H:25])[N+:14](=[O:16])[O-:15])[C:5]([H:20])([H:21])[S:2](=[O:3])(=[O:4])[C:1]([H:17])([H:18])[H:19]"
    ),
    # Issue 119
    (
        r"[H:1][C:2]([H:3])([H:4])[c:5]1[c:6]2=[N:7][O:8][N+:9](=[c:10]2[c:11]([n+:12]([n+:13]1[O-:14])[O-:15])[C:16]([H:17])([H:18])[H:19])[O-:20]",
        r"[H:1][C:2]([H:3])([H:4])[c:5]1[c:6]2=[N:7][O:8][N+:9](=[c:10]2[c:11]([n:12](=[O:15])[n:13]1=[O:14])[C:16]([H:17])([H:18])[H:19])[O-:20]",
    ),
    (
        r"[H:1][c:2]1[c:3]([c:4]([c:5]2[c:6]([c:7]1[H:8])/[C:9](=[N:10]/[C:11](=[O:12])[c:13]3[c:14]([c:15]([c:16]([c:17]([c:18]3[N+:19](=[O:20])[O-:21])[H:22])[N+:23](=[O:24])[O-:25])[H:26])[H:27])/[N-:28][c:29]4[c:30]([c:31]([c:32]([c:33]([n+:34]4[C:35]2([H:36])[H:37])[H:38])[Br:39])[H:40])[H:41])[H:42])[H:43]",
        r"[H:1][c:2]1[c:3]([c:4]([c:5]2[c:6]([c:7]1[H:8])/[C:9](=[N:10]/[C:11](=[O:12])[c:13]3[c:14]([c:15]([c:16]([c:17]([c:18]3[N+:19](=[O:20])[O-:21])[H:22])[N+:23](=[O:24])[O-:25])[H:26])[H:27])/[N:28]=[C:29]4[C:30](=[C:31]([C:32](=[C:33]([N:34]4[C:35]2([H:36])[H:37])[H:38])[Br:39])[H:40])[H:41])[H:42])[H:43]"
    ),
    (
        r"[H:21][c:1]1[c:2]([c:3]([c:4]([c:5]([c:6]1[C:7]2=[N:8][N+:9]3=[C:15]([S:16]2)[N:14]([C:12](=[O:13])[C:11](=[C:10]3[O-:17])[H:25])[H:26])[H:24])[H:23])[N:18](=[O:19])=[O:20])[H:22]",
        r"[H:21][c:1]1[c:2]([c:3]([c:4]([c:5]([c:6]1[C:7]2=[N:8][N:9]3[C:10](=[C:11]([C:12](=[O:13])[N+:14](=[C:15]3[S:16]2)[H:26])[H:25])[O-:17])[H:24])[H:23])[N+:18](=[O:20])[O-:19])[H:22]"
    )

]

@pytest.mark.skipif(not OPENEYE_AVAILABLE, reason="requires openeye")
@pytest.mark.parametrize(
    "given_smiles, expected_smiles",
    NORMALIZATION_MOLECULE_TESTS
)
def test_normalize_molecule_openeye(given_smiles, expected_smiles):
    from openff.toolkit.topology.molecule import Molecule
    expected_molecule = Molecule.from_mapped_smiles(expected_smiles)

    molecule = Molecule.from_mapped_smiles(given_smiles)
    assert not Molecule.are_isomorphic(molecule, expected_molecule)[0]

    output_molecule = normalize_molecule(molecule)
    output_smiles = output_molecule.to_smiles(mapped=True)
    # reload molecule to avoid spurious failures from different kekulization
    output_molecule = Molecule.from_mapped_smiles(output_smiles)
    is_isomorphic = Molecule.are_isomorphic(
        output_molecule, expected_molecule,
    )[0]
    assert is_isomorphic, output_molecule.to_smiles(mapped=True)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "given_smiles, expected_smiles",
    NORMALIZATION_MOLECULE_TESTS
)
def test_normalize_molecule_bypasses_rdkit_normalization(
    given_smiles,
    expected_smiles,
):
    from openff.toolkit.topology.molecule import Molecule

    expected_molecule = _load_rdkit_molecule_exactly(expected_smiles)
    molecule = _load_rdkit_molecule_exactly(given_smiles)
    
    assert not Molecule.are_isomorphic(molecule, expected_molecule)[0]
    output_molecule = normalize_molecule(molecule)
    is_isomorphic = Molecule.are_isomorphic(output_molecule, expected_molecule)[0]

    # this may fail spuriously due to kekulization error, in which case
    # we reload the molecule and try again
    if not is_isomorphic:
        output_smiles = output_molecule.to_smiles(mapped=True)
        # reload molecule to avoid spurious failures from different kekulization
        output_molecule = _load_rdkit_molecule_exactly(output_smiles)
        is_isomorphic = Molecule.are_isomorphic(
            output_molecule, expected_molecule,
        )[0]

    assert is_isomorphic, output_molecule.to_smiles(mapped=True)


        


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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=MultipleComponentsInMoleculeWarning,
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


@pytest.mark.skipif(not RDKIT_AVAILABLE or not OPENEYE_AVAILABLE, reason="requires rdkit and openeye")
@pytest.mark.parameterize(
    "toolkit_combinations",
    [
        [RDKitToolkitWrapper()],
        [RDKitToolkitWrapper(), OpenEyeToolkitWrapper()], # check precedence
    ]
)
def test_toolkit_registry_passes_through_nagl(toolkit_combinations):
    """
    Tests issue #177: OpenEye being called when disallowed by the native toolkit registry manager
    """

    from rdkit.Chem import ForwardSDMolSupplier
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper


    suppl = ForwardSDMolSupplier(gzip.open(COFACTOR_SDF_GZ), removeHs=False)
    rdmol = list(suppl)[0]
    m = Molecule.from_rdkit(rdmol)

    # Force AmberTools + RDKit
    amber_rdkit = ToolkitRegistry([*toolkit_combinations])

    with toolkit_registry_manager(amber_rdkit):
        m.assign_partial_charges(
            partial_charge_method="openff-gnn-am1bcc-0.1.0-rc.1.pt",
            toolkit_registry=NAGLToolkitWrapper(),
        )
