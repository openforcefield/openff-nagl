import pytest

from openff.nagl.app.partition import DatasetPartitioner

class TestDatasetPartitioner:

    @pytest.mark.parametrize(
        "additional_input, additional_output",
        [
            ([], []),
            (["CCCC"], []),
            (["CCCC", "CCCCCCCCN", "CCCCCCCCCN", "CCCCCCCCCCCN"], ["CCCCCCCCCN", "CCCCCCCCCCCN"])
        ]
    )
    def test_from_smiles(self, additional_input, additional_output):
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
        expected_smiles = set(base_input + additional_output)

        partitioner = DatasetPartitioner.from_smiles(all_smiles)
        selected_smiles = partitioner.select_molecules(
            n_environment_molecules=2,
        )
        assert set(selected_smiles) == expected_smiles