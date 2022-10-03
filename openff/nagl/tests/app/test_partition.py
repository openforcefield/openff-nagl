import pytest

from openff.nagl.app.partition import DatasetPartitioner

class TestDatasetPartitioner:

    @pytest.mark.parametrize(
        "additional_input, n_additional_output",
        [
            ([], 0),
            (["CCCC"], 0),
            (["CCCC", "CCCCCCCCN", "CCCCCCCCCN", "CCCCCCCCCCCN"], 2)
        ]
    )
    def test_from_smiles(self, additional_input, n_additional_output):
        base_input = [
            'C=C', 'C=CC', 'C=CCC', 'C=CCCC',
            'C#C', 'C#CC', 'C#CCC',
            'CF', 'CCF', 'CCCF', "CCCCF", 'FCCCCl',
            'O', 'CO', 'CCO', 'CCCO', 'CCCCCO',
            'C', 'CC', 'CCC',
            'S',
            'CCC(=O)O', 'C=O',
            'CCCCOCCCCC', 'CCCCCCCCCCOCCCC',
        ]
        all_smiles = base_input + additional_input
        expected_smiles = set(base_input)

        partitioner = DatasetPartitioner.from_smiles(all_smiles)
        selected_smiles = partitioner.select_molecules(
            n_environment_molecules=2,
        )
        assert expected_smiles.issubset(selected_smiles)
        n_difference = len(set(selected_smiles) - expected_smiles)
        assert n_difference == n_additional_output


    def test_iadd(self):
        partitioner = DatasetPartitioner.from_smiles(["C"])
        assert len(partitioner.environments_by_element) == 2
        assert len(partitioner.environments_by_element["C"]) == 1
        assert len(partitioner.environments_by_element["H"]) == 1
        assert len(partitioner.molecule_atom_fps) == 1

        assert partitioner._all_environments is None
        assert len(partitioner.all_environments) == 2
        assert partitioner.all_environments is partitioner._all_environments


        partitioner += DatasetPartitioner.from_smiles(["S"])
        assert len(partitioner.environments_by_element) == 3
        assert len(partitioner.environments_by_element["C"]) == 1
        assert len(partitioner.environments_by_element["H"]) == 2
        assert len(partitioner.environments_by_element["S"]) == 1
        assert len(partitioner.molecule_atom_fps) == 2

        assert partitioner._all_environments is None
        assert len(partitioner.all_environments) == 4