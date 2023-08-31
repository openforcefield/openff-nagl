import pytest
from openff.toolkit.topology import Molecule
from openff.units import unit

from openff.nagl.tests.testing.utils import assert_smiles_equal
from openff.nagl.utils.resonance import FragmentEnumerator, ResonanceEnumerator


@pytest.fixture
def resonance_enumerator():
    explicit_smarts = "[O-:1][N+:2](=[O:3])[N:4](-[H:16])[c:5]1[c:6](-[H:12])[c:7](-[H:13])[c:8](-[H:14])[c:9](-[H:15])[n+:10]1[O-:11]"
    offmol = Molecule.from_mapped_smiles(explicit_smarts)
    enumerator = ResonanceEnumerator(offmol)
    return enumerator


class TestFragmentEnumerator:
    @pytest.fixture
    def fragment_enumerator(self, resonance_enumerator):
        enumerator = FragmentEnumerator(resonance_enumerator.reduced_graph)

        # ensure a particular kekulization
        aromatic_bonds = {
            (4, 5): 2,
            (5, 6): 1,
            (6, 7): 2,
            (7, 8): 1,
            (8, 9): 2,
            (4, 9): 1,
        }

        for (i, j), bond_order in aromatic_bonds.items():
            enumerator.reduced_graph.edges[i, j]["bond_order"] = bond_order

        return enumerator

    def test_creation(self, fragment_enumerator):
        atomic_numbers = [8, 7, 8, 7, 6, 6, 6, 6, 6, 7, 8]
        fragment_atomic_numbers = [
            z for _, z in fragment_enumerator.reduced_graph.nodes(data="atomic_number")
        ]
        assert fragment_atomic_numbers == atomic_numbers

        formal_charges = [-1, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1]
        fragment_formal_charges = [
            int(q.m_as(unit.elementary_charge))
            for _, q in fragment_enumerator.reduced_graph.nodes(data="formal_charge")
        ]
        assert fragment_formal_charges == formal_charges

    def test_get_resonance_types(self, fragment_enumerator):
        assert fragment_enumerator.donor_indices == [0, 3, 10]
        assert fragment_enumerator.acceptor_indices == [1, 2, 9]

    @pytest.mark.parametrize(
        "donor, acceptor, expected_paths",
        [
            (0, 1, tuple()),
            (0, 2, ((0, 1, 2),)),
            (0, 9, ((0, 1, 3, 4, 5, 6, 7, 8, 9), (0, 1, 3, 4, 9))),
            (10, 1, ((10, 9, 8, 7, 6, 5, 4, 3, 1), (10, 9, 4, 3, 1))),
            (10, 2, tuple()),
            (10, 9, tuple()),
        ],
    )
    def test_get_donor_to_acceptor_paths(
        self,
        fragment_enumerator,
        donor,
        acceptor,
        expected_paths,
    ):
        paths = fragment_enumerator._get_all_odd_n_simple_paths(donor, acceptor)
        assert paths == expected_paths

    @pytest.mark.parametrize(
        "path, is_transfer_path",
        [
            ((0, 1, 2), True),
            ((0, 1, 3, 4, 9), False),
            ((0, 1, 3, 4, 5, 6, 7, 8, 9), False),
            ((10, 9, 4, 3, 1), False),
            ((10, 9, 8, 7, 6, 5, 4, 3, 1), False),
        ],
    )
    def test_is_transfer_path(self, fragment_enumerator, path, is_transfer_path):
        assert fragment_enumerator._is_transfer_path(path) == is_transfer_path

    @pytest.mark.parametrize(
        "acceptor, donor, expected_paths",
        [
            (1, 0, []),
            (1, 10, []),
            (1, 3, []),
            (2, 0, [(0, 1, 2)]),
            (2, 10, []),
            (2, 3, [(3, 1, 2)]),
            (9, 0, []),
            (9, 10, []),
            (9, 3, [(3, 4, 5, 6, 7, 8, 9)]),
        ],
    )
    def test_get_transfer_paths(
        self, fragment_enumerator, acceptor, donor, expected_paths
    ):
        paths = fragment_enumerator._get_transfer_paths(donor, acceptor)
        assert list(paths) == expected_paths

    def test_to_resonance_dict(self, fragment_enumerator):
        expected = {
            "acceptor_indices": [1, 2, 9],
            "donor_indices": [0, 3, 10],
            "bond_orders": {
                (0, 1): 1,
                (1, 2): 2,
                (1, 3): 1,
                (3, 4): 1,
                (4, 5): 2,
                (4, 9): 1,
                (5, 6): 1,
                (6, 7): 2,
                (7, 8): 1,
                (8, 9): 2,
                (9, 10): 1,
            },
            "formal_charges": [-1, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1],
        }
        given = fragment_enumerator._to_resonance_dict(include_formal_charges=True)
        assert given == expected

    def test_enumerate_donor_acceptor_resonance_forms(self, fragment_enumerator):
        fragments = list(
            fragment_enumerator._enumerate_donor_acceptor_resonance_forms()
        )
        assert len(fragments) == 3

        # (0, 1, 2)
        resonance_fragment1 = fragments[0]._to_resonance_dict(
            include_formal_charges=True
        )
        expected_fragment1 = {
            "acceptor_indices": [0, 1, 9],
            "donor_indices": [2, 3, 10],
            "bond_orders": {
                (0, 1): 2,
                (1, 2): 1,
                (1, 3): 1,
                (3, 4): 1,
                (4, 5): 2,
                (4, 9): 1,
                (5, 6): 1,
                (6, 7): 2,
                (7, 8): 1,
                (8, 9): 2,
                (9, 10): 1,
            },
            "formal_charges": [0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1],
        }
        assert resonance_fragment1 == expected_fragment1

        # (3, 1, 2)
        resonance_fragment2 = fragments[1]._to_resonance_dict(
            include_formal_charges=True
        )
        expected_fragment2 = {
            "acceptor_indices": [1, 3, 9],
            "donor_indices": [0, 2, 10],
            "bond_orders": {
                (0, 1): 1,
                (1, 2): 1,
                (1, 3): 2,
                (3, 4): 1,
                (4, 5): 2,
                (4, 9): 1,
                (5, 6): 1,
                (6, 7): 2,
                (7, 8): 1,
                (8, 9): 2,
                (9, 10): 1,
            },
            "formal_charges": [-1, 1, -1, 1, 0, 0, 0, 0, 0, 1, -1],
        }
        assert resonance_fragment2 == expected_fragment2

        # (3, 4, 5, 6, 7, 8, 9)
        resonance_fragment3 = fragments[2]._to_resonance_dict(
            include_formal_charges=True
        )
        expected_fragment3 = {
            "acceptor_indices": [1, 2, 3],
            "donor_indices": [0, 9, 10],
            "bond_orders": {
                (0, 1): 1,
                (1, 2): 2,
                (1, 3): 1,
                (3, 4): 2,
                (4, 5): 1,
                (4, 9): 1,
                (5, 6): 2,
                (6, 7): 1,
                (7, 8): 2,
                (8, 9): 1,
                (9, 10): 1,
            },
            "formal_charges": [-1, 1, 0, 1, 0, 0, 0, 0, 0, 0, -1],
        }
        assert resonance_fragment3 == expected_fragment3

    @pytest.mark.parametrize(
        "path, expected_bonds",
        [
            ([], [1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1]),
            ([3, 1, 2], [1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1]),
        ],
    )
    def test_transfer_electrons(self, fragment_enumerator, path, expected_bonds):
        transferred = fragment_enumerator._transfer_electrons(path)
        resonance = transferred._to_resonance_dict()["bond_orders"]
        new_bonds = list(resonance.values())
        assert new_bonds == expected_bonds

    def test_select_lowest_energy_forms(self):
        input_smiles = {
            "lowest": "[N:1]([H:2])([H:3])[C:4](=[O:5])[H:6]",
            "not_lowest": "[N+:1]([H:2])([H:3])=[C:4]([O-:5])[H:6]",
        }
        choices = {}
        for k, smi in input_smiles.items():
            offmol = Molecule.from_mapped_smiles(smi)
            choices[k] = ResonanceEnumerator(offmol).to_fragment()

        lowest = FragmentEnumerator._select_lowest_energy_forms(choices)
        assert set(lowest) == {"lowest"}


class TestFragmentEnumeratorCarboxylate:
    @pytest.fixture()
    def enumerator(self):
        molecule = Molecule.from_mapped_smiles("[C:1]([O-:2])(=[O:3])([H:4])")
        res = ResonanceEnumerator(molecule)
        return res.to_fragment()

    def test_as_transferred(self, enumerator):
        original = enumerator._to_resonance_dict(include_formal_charges=True)

        assert original["formal_charges"] == [0, -1, 0]
        assert original["bond_orders"] == {(0, 1): 1, (0, 2): 2}

        path = (1, 0, 2)
        transferred = enumerator._transfer_electrons(path)
        transferred_info = transferred._to_resonance_dict(include_formal_charges=True)

        assert transferred_info["formal_charges"] == [0, 0, -1]
        assert transferred_info["bond_orders"] == {(0, 1): 2, (0, 2): 1}

    def test_enumerate_donor_acceptor_resonance_forms(self, enumerator):
        forms = list(enumerator._enumerate_donor_acceptor_resonance_forms())
        assert len(forms) == 1

        info = forms[0]._to_resonance_dict(include_formal_charges=True)
        assert info["formal_charges"] == [0, 0, -1]
        assert info["bond_orders"] == {(0, 1): 2, (0, 2): 1}


class TestResonanceEnumerator:
    expected_resonance_smiles = [
        "[H]C1=C(C(=[N+]([H])N([O-])[O-])[N+](=O)C(=C1[H])[H])[H]",
        "[H]C1=C(C(=[N+]([H])[N+](=O)[O-])N(C(=C1[H])[H])[O-])[H]",
        "[H]c1c(c([n+](c(c1[H])N([H])[N+](=O)[O-])[O-])[H])[H]",
        "[H]c1c(c([n+](c(c1[H])[N+](=[N+]([O-])[O-])[H])[O-])[H])[H]",
    ]

    @pytest.mark.parametrize(
        "lowest_energy_only, include_all_transfer_pathways, n_expected, expected_smiles",
        [
            (True, True, 7, expected_resonance_smiles[:3]),
            (True, False, 5, expected_resonance_smiles[:3]),
            (False, True, 9, expected_resonance_smiles),
            (False, False, 6, expected_resonance_smiles),
        ],
    )
    def test_enumerate_resonance_forms_multiple(
        self,
        resonance_enumerator,
        lowest_energy_only,
        include_all_transfer_pathways,
        n_expected,
        expected_smiles,
    ):
        from openff.nagl.tests.testing import utils

        resonance_forms = resonance_enumerator.enumerate_resonance_forms(
            lowest_energy_only=lowest_energy_only,
            include_all_transfer_pathways=include_all_transfer_pathways,
            as_dicts=False,
        )
        smiles = {form.to_smiles() for form in resonance_forms}

        expected_smiles = {utils.clean_smiles(smi) for smi in expected_smiles}

        assert smiles == expected_smiles
        assert len(resonance_forms) == n_expected

    @pytest.mark.parametrize("lowest_energy_only", [True, False])
    @pytest.mark.parametrize("include_all_transfer_pathways", [True, False])
    def test_enumerate_resonance_forms_simple(
        self, lowest_energy_only, include_all_transfer_pathways
    ):
        methane = Molecule.from_smiles("C")
        enumerator = ResonanceEnumerator(methane)
        resonance_molecules = enumerator.enumerate_resonance_forms(
            lowest_energy_only=lowest_energy_only,
            include_all_transfer_pathways=include_all_transfer_pathways,
        )
        assert len(resonance_molecules) == 1

        output_smiles = resonance_molecules[0].to_smiles()
        assert_smiles_equal(output_smiles, "C")

    @pytest.mark.parametrize(
        "smiles, expected_indices",
        [
            (
                "[C:1](-[H:2])(-[H:3])(-[H:4])(-[H:5])",
                [],
            ),
            (
                "[C:1](=[O:4])([O-:5])[C:2]([H:8])([H:9])[C:3](=[O:6])([O-:7])",
                [(0, 3, 4), (2, 5, 6)],
            ),
            (
                "[C:1](=[O:3])([O-:4])[C:2](=[O:5])([O-:6])",
                [(0, 1, 2, 3, 4, 5)],
            ),
            (
                "[O-:1][N+:2](=[O:3])[N:4](-[H:16])[c:5]1[c:6](-[H:12])[c:7](-[H:13])[c:8](-[H:14])[c:9](-[H:15])[n+:10]1[O-:11]",
                [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)],
            ),
        ],
    )
    def test_get_acceptor_donor_fragments(self, smiles, expected_indices):
        offmol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        enumerator = ResonanceEnumerator(offmol)
        fragments = enumerator._get_acceptor_donor_fragments()

        fragment_indices = sorted(
            tuple(fragment.reduced_graph.nodes) for fragment in fragments
        )
        assert fragment_indices == expected_indices
