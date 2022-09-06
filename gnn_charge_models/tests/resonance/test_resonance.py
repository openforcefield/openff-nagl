import pytest
from typing import List

from openff.toolkit.topology.molecule import Molecule as OFFMolecule
from gnn_charge_models.resonance.resonance import ResonanceEnumerator, FragmentEnumerator
from gnn_charge_models.tests.testing import utils


class TestEnumerateResonanceForms:
    expected_resonance_smiles = [
        "[H]C1=C(C(=[N+]([H])N([O-])[O-])[N+](=O)C(=C1[H])[H])[H]",
        "[H]C1=C(C(=[N+]([H])[N+](=O)[O-])N(C(=C1[H])[H])[O-])[H]",
        "[H]c1c(c([n+](c(c1[H])N([H])[N+](=O)[O-])[O-])[H])[H]",
        "[H]c1c(c([n+](c(c1[H])[N+](=[N+]([O-])[O-])[H])[O-])[H])[H]",
    ]

    @pytest.fixture
    def fragment_enumerator(self):
        offmol = OFFMolecule.from_mapped_smiles(
            "[O-:1][N+:2](=[O:3])[N:4][c:5]1[c:6][c:7][c:8][c:9][n+:10]1[O-:11]",
            allow_undefined_stereo=True,
        )
        rdmol = offmol.to_rdkit()
        return FragmentEnumerator(rdmol, clean_molecule=True)

    def test_get_resonance_types(self, fragment_enumerator):
        assert fragment_enumerator.acceptor_indices == [1, 2, 9]
        assert fragment_enumerator.donor_indices == [0, 3, 10]

    @pytest.mark.parametrize(
        "donor, acceptor, expected_paths",
        [
            (0, 1, tuple()),
            (0, 2, ((0, 1, 2),)),
            (0, 9, ((0, 1, 3, 4, 5, 6, 7, 8, 9), (0, 1, 3, 4, 9))),
            (10, 1, ((10, 9, 4, 3, 1),
                     (10, 9, 8, 7, 6, 5, 4, 3, 1))),
            (10, 2, tuple()),
            (10, 9, tuple()),
        ]
    )
    def test_get_donor_to_acceptor_paths(
        self, fragment_enumerator,
        donor, acceptor, expected_paths,
    ):
        paths = fragment_enumerator.path_generator.all_odd_node_simple_paths(
            donor, acceptor)
        assert paths == expected_paths

    @pytest.mark.parametrize(
        "path, is_transfer_path",
        [
            ((0, 1, 2), True),
            ((0, 1, 3, 4, 9), False),
            ((0, 1, 3, 4, 5, 6, 7, 8, 9), False),
            ((10, 9, 4, 3, 1), False),
            ((10, 9, 8, 7, 6, 5, 4, 3, 1), False)

        ]
    )
    def test_is_transfer_path(self, fragment_enumerator, path, is_transfer_path):
        assert fragment_enumerator.is_transfer_path(path) == is_transfer_path

    @pytest.mark.parametrize(
        "lowest_energy_only, include_all_transfer_pathways, n_expected, expected_smiles",
        [
            (True, True, 7, expected_resonance_smiles[:3]),
            (True, False, 5, expected_resonance_smiles[:3]),
            (False, True, 9, expected_resonance_smiles),
            (False, False, 6, expected_resonance_smiles),
        ]
    )
    def test_enumerate_resonance_forms(
        self,
        fragment_enumerator,
        lowest_energy_only,
        include_all_transfer_pathways,
        n_expected,
        expected_smiles,
    ):
        fragments = fragment_enumerator.enumerate_resonance_forms(
            lowest_energy_only=lowest_energy_only,
            include_all_transfer_pathways=include_all_transfer_pathways,
        )
        smiles = {
            utils.rdkit_molecule_to_smiles(fragment.rdkit_molecule)
            for fragment in fragments.values()
        }

        expected_smiles = {
            utils.clean_smiles(smi)
            for smi in expected_smiles
        }

        assert smiles == expected_smiles
        assert len(fragments) == n_expected

    def test_to_resonance_dict(self, fragment_enumerator):
        expected = {
            'acceptor_indices': [1, 2, 9],
            'donor_indices': [0, 3, 10],
            'bond_orders': {0: 1, 1: 2, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 7: 2, 8: 1, 9: 2, 10: 1},
            'formal_charges': [-1, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1],
        }
        given = fragment_enumerator.to_resonance_dict(
            include_formal_charges=True)
        assert given == expected

    def test_enumerate_donor_acceptor_resonance_forms(self, fragment_enumerator):
        fragments = list(
            fragment_enumerator.enumerate_donor_acceptor_resonance_forms())
        assert len(fragments) == 3
        assert fragments[0].to_resonance_dict(include_formal_charges=True) == {
            'acceptor_indices': [0, 1, 9],
            'donor_indices': [2, 3, 10],
            'bond_orders': {0: 2, 1: 1, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 7: 2, 8: 1, 9: 2, 10: 1},
            'formal_charges': [0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1],
        }
        assert fragments[1].to_resonance_dict(include_formal_charges=True) == {
            'acceptor_indices': [1, 3, 9],
            'donor_indices': [0, 2, 10],
            'bond_orders': {0: 1, 1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 1, 7: 2, 8: 1, 9: 2, 10: 1},
            'formal_charges': [-1, 1, -1, 1, 0, 0, 0, 0, 0, 1, -1],
        }
        assert fragments[2].to_resonance_dict(include_formal_charges=True) == {
            'acceptor_indices': [1, 2, 3],
            'donor_indices': [0, 9, 10],
            'bond_orders': {0: 1, 1: 2, 2: 1, 3: 2, 4: 1, 5: 1, 6: 2, 7: 1, 8: 2, 9: 1, 10: 1},
            'formal_charges': [-1, 1, 0, 1, 0, 0, 0, 0, 0, 0, -1],
        }

    @pytest.mark.parametrize(
        "acceptor, donor, expected_paths",
        [
            (1, 0, []),
            (1, 10,  []),
            (1, 3, []),
            (2, 0, [(0, 1, 2)]),
            (2, 10,  []),
            (2, 3, [(3, 1, 2)]),
            (9, 0, []),
            (9, 10,  []),
            (9, 3, [(3, 4, 5, 6, 7, 8, 9)]),
        ]
    )
    def test_get_transfer_paths(self, fragment_enumerator, acceptor, donor, expected_paths):
        paths = fragment_enumerator.get_transfer_paths(donor, acceptor)
        assert list(paths) == expected_paths

    @pytest.mark.parametrize(
        "path, expected_bonds",
        [
            ([0, 1, 2], [2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1]),
        ]
    )
    def test_as_transferred(self, fragment_enumerator, path, expected_bonds):
        original = list(fragment_enumerator.get_bond_orders().values())
        assert original == [1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1]
        as_transferred = fragment_enumerator.as_transferred(path)
        new_bond_orders = list(as_transferred.get_bond_orders().values())
        assert new_bond_orders == expected_bonds


class TestResonanceEnumeratorCase:

    @pytest.fixture
    def enumerator(self):
        return ResonanceEnumerator.from_smiles(
            "[O-][N+](=O)Nc1cccc[n+]1[O-]",
            mapped=False
        )

    def test_select_acceptor_donor_fragments(self, enumerator):
        assert enumerator.rdkit_molecule.GetNumAtoms() == 16
        enumerator.select_acceptor_donor_fragments()
        assert len(enumerator.acceptor_donor_fragments) == 1

        fragment = enumerator.acceptor_donor_fragments[0]
        assert fragment.rdkit_molecule.GetNumAtoms() == 11

        smiles = utils.rdkit_molecule_to_smiles(fragment.rdkit_molecule)
        expected_smiles = "[H]c1c(c([n+](c(c1[H])N([H])[N+](=O)[O-])[O-])[H])[H]"
        expected_smiles = utils.clean_smiles(expected_smiles)
        assert smiles == expected_smiles


class TestResonanceEnumeratorMethane:

    @pytest.fixture
    def enumerator(self):
        return ResonanceEnumerator.from_smiles("C")

    @pytest.mark.parametrize("lowest_energy_only", [True, False])
    @pytest.mark.parametrize("include_all_transfer_pathways", [True, False])
    def test_enumerate_resonance_molecules(self, enumerator, lowest_energy_only, include_all_transfer_pathways):
        rdmols = enumerator.enumerate_resonance_molecules(
            lowest_energy_only=lowest_energy_only,
            include_all_transfer_pathways=include_all_transfer_pathways,
        )
        assert len(rdmols) == 1

        smiles = utils.rdkit_molecule_to_smiles(rdmols[0])
        expected_smiles = utils.clean_smiles("C")
        assert smiles == expected_smiles


@pytest.mark.parametrize(
    "smiles, expected_indices",
    [
        (
            "[C:1](-[H:2])(-[H:3])(-[H:4])",
            [],
        ),
        (
            "[C:1](=[O:4])([O-:5])[C:2]([H:8])([H:9])[C:3](=[O:6])([O-:7])",
            [(0, 3, 4), (2, 5, 6)]
        ),
        (
            "[C:1](=[O:3])([O-:4])[C:2](=[O:5])([O-:6])",
            [(0, 1, 2, 3, 4, 5)],
        ),
        (
            "[O-:1][N+:2](=[O:3])[N:4][c:5]1[c:6][c:7][c:8][c:9][n+:10]1[O-:11]",
            [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)],
        )
    ]
)
def test_select_acceptor_donor_fragments(smiles, expected_indices):
    enumerator = ResonanceEnumerator.from_smiles(smiles, mapped=True)
    enumerator.select_acceptor_donor_fragments()
    fragment_indices = sorted([
        tuple(sorted(fragment.original_to_current_atom_indices))
        for fragment in enumerator.acceptor_donor_fragments
    ])

    assert fragment_indices == expected_indices


def test_select_lowest_energy_forms():
    input_smiles = {
        "lowest": "[N:1]([H:2])([H:3])[C:4](=[O:5])[H:6]",
        "not_lowest": "[N+:1]([H:2])([H:3])=[C:4]([O-:5])[H:6]",
    }
    choices = {}
    for k, smi in input_smiles.items():
        rdmol = OFFMolecule.from_mapped_smiles(smi).to_rdkit()
        choices[k] = FragmentEnumerator(rdmol)

    lowest = FragmentEnumerator.select_lowest_energy_forms(choices)
    assert set(lowest) == {"lowest"}


class TestFragmentEnumeratorCarboxylate:

    @pytest.fixture()
    def enumerator(self):
        res = ResonanceEnumerator.from_smiles(
            "[C:1]([O-:2])(=[O:3])([H:4])",
            mapped=True
        )
        res.select_acceptor_donor_fragments()
        return res.acceptor_donor_fragments[0]

    def test_as_transferred(self, enumerator):
        original_charges = enumerator.get_formal_charges()
        assert original_charges == [0, -1, 0]

        original_bonds = list(enumerator.get_bond_orders().values())
        assert original_bonds == [1, 2]

        path = (1, 0, 2)
        transferred = enumerator.as_transferred(path)

        transferred_charges = transferred.get_formal_charges()
        assert transferred_charges == [0, 0, -1]

        transferred_bonds = transferred.get_bond_orders().values()
        assert list(transferred_bonds) == [2, 1]

    def test_get_donor_acceptance_resonance_forms(self, enumerator):
        forms = list(enumerator.get_donor_acceptance_resonance_forms(1, 2))
        assert len(forms) == 1

        assert forms[0].get_formal_charges() == [0, 0, -1]
        assert list(forms[0].get_bond_orders().values()) == [2, 1]

    def test_enumerate_donor_acceptor_resonance_forms(self, enumerator):
        forms = list(enumerator.enumerate_donor_acceptor_resonance_forms())
        assert len(forms) == 1

        assert forms[0].get_formal_charges() == [0, 0, -1]
        assert list(forms[0].get_bond_orders().values()) == [2, 1]
