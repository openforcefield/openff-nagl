# import pytest

# from openff.nagl.app.partition import DatasetPartitioner

# class TestDatasetPartitioner:

#     @pytest.mark.parametrize(
#         "additional_input, n_additional_output",
#         [
#             ([], 0),
#             (["CCCC"], 0),
#             (["CCCC", "CCCCCCCCN", "CCCCCCCCCN", "CCCCCCCCCCCN"], 2)
#         ]
#     )
#     def test_from_smiles(self, additional_input, n_additional_output):
#         base_input = [
#             'C=C', 'C=CC', 'C=CCC', 'C=CCCC',
#             'C#C', 'C#CC', 'C#CCC',
#             'CF', 'CCF', 'CCCF', "CCCCF", 'FCCCCl',
#             'O', 'CO', 'CCO', 'CCCO', 'CCCCCO',
#             'C', 'CC', 'CCC',
#             'S',
#             'CCC(=O)O', 'C=O',
#             'CCCCOCCCCC', 'CCCCCCCCCCOCCCC',
#         ]
#         all_smiles = base_input + additional_input
#         expected_smiles = set(base_input)

#         partitioner = DatasetPartitioner.from_smiles(all_smiles)
#         selected_smiles = partitioner.select_molecules(
#             n_environment_molecules=2,
#         )
#         assert expected_smiles.issubset(selected_smiles)
#         n_difference = len(set(selected_smiles) - expected_smiles)
#         assert n_difference == n_additional_output


#     def test_iadd(self):
#         partitioner = DatasetPartitioner.from_smiles(["C"])
#         assert len(partitioner.environments_by_element) == 2
#         assert len(partitioner.environments_by_element["C"]) == 1
#         assert len(partitioner.environments_by_element["H"]) == 1
#         assert len(partitioner.molecule_atom_fps) == 1

#         assert partitioner._all_environments is None
#         assert len(partitioner.all_environments) == 2
#         assert partitioner.all_environments is partitioner._all_environments


#         partitioner += DatasetPartitioner.from_smiles(["S"])
#         assert len(partitioner.environments_by_element) == 3
#         assert len(partitioner.environments_by_element["C"]) == 1
#         assert len(partitioner.environments_by_element["H"]) == 2
#         assert len(partitioner.environments_by_element["S"]) == 1
#         assert len(partitioner.molecule_atom_fps) == 2

#         assert partitioner._all_environments is None
#         assert len(partitioner.all_environments) == 4


#     @pytest.mark.parametrize(
#         "n, input_fractions, expected_counts",
#         [
#             (10, (0.5, 0.5, 0), (5, 5, 0)),
#             (10, (0.5, 0.25, 0.25), (6, 2, 2)),
#             (10, (0.7, 0.2, 0.1), (7, 2, 1)),
#         ]
#     )
#     def test_get_counts_from_fractions(self, n, input_fractions, expected_counts):
#         counts = DatasetPartitioner._get_counts_from_fractions(n, *input_fractions)
#         assert counts == expected_counts

#     def test_flatten_environment(self):
#         smiles = [
#             'C=C', 'C=CC', 'C=CCC', 'C=CCCC',
#             'C#C', 'C#CC', 'C#CCC',
#             'CF', 'CCF', 'CCCF', "CCCCF", 'FCCCCl',
#             'O', 'CO', 'CCO', 'CCCO', 'CCCCCO',
#             'C', 'CC', 'CCC', 'CCCC', 'CCCCCCC',
#             'S',
#             'CCC(=O)O', 'C=O',
#             'CCCCOCCCCC', 'CCCCCCCCCCOCCCC',
#         ]
#         partitioner = DatasetPartitioner.from_smiles(smiles)
#         environments = partitioner._get_flattened_smiles_environments()
#         expected_counts = [1, 5, 1, 1, 1, 1, 9, 9]

#         counts = [len(x) for x in environments]
#         assert len(counts) == 10
#         assert counts[:8] == expected_counts
#         assert set(counts[8:]) == {25, 27}

    
#     def test_partition(self):
#         smiles = [
#             'C=C', 'C=CC', 'C=CCC', 'C=CCCC',
#             'C#C', 'C#CC', 'C#CCC',
#             'CF', 'CCF', 'CCCF', "CCCCF", 'FCCCCl',
#             'O', 'CO', 'CCO', 'CCCO', 'CCCCCO',
#             'C', 'CC', 'CCC', 'CCCC', 'CCCCCCC',
#             'S',
#             'CCC(=O)O', 'C=O',
#             'CCCCOCCCCC', 'CCCCCCCCCCOCCCC',
#         ]
#         partitioner = DatasetPartitioner.from_smiles(smiles)
#         expected_element_counts = {
#             "C": [1] * 29 + [2] * 5 + [3] * 3 + [4, 6, 10],
#             "H": [1] * 11 + [3] * 3 + [4] * 3 + [5, 11, 16],
#             "O": [1] * 5 + [2, 3],
#             "S": [1],
#             "F": [1, 4],
#             "Cl": [1],
#         }

#         assert len(partitioner.environments_by_element) == len(expected_element_counts)
#         for el, expected_counts in expected_element_counts.items():
#             env_dict = partitioner.environments_by_element[el]
#             counts = sorted([len(x) for x in env_dict.values()])
#             assert counts == expected_counts


#         train, val, test = partitioner.partition(
#             training_fraction=0.7,
#             validation_fraction=0.2,
#             test_fraction=0.1,
#         )

#         assert len(train) == 20
#         assert len(val) == 5
#         assert len(test) == 2
        

#         training_partition = DatasetPartitioner.from_smiles(train)
#         expected_training_element_counts = {
#             "C": [1] * 28 + [2] * 6 + [3, 3, 5],
#             "H": [1] * 10 + [3] * 6 + [4, 6, 10],
#             "O": [1] * 5 + [2],
#             "S": [1],
#             "F": [1, 3],
#             "Cl": [1],
#         }

#         assert len(training_partition.environments_by_element) == len(expected_training_element_counts)
#         for el, expected_counts in expected_training_element_counts.items():
#             env_dict = training_partition.environments_by_element[el]
#             counts = sorted([len(x) for x in env_dict.values()])
#             assert counts == expected_counts
        

#         val_partition = DatasetPartitioner.from_smiles(val)
#         expected_val_element_counts = {
#             "C": [1] * 7 + [3],
#             "H": [1] * 4 + [3, 4],
#             "O": [1, 1]
#         }

#         for el, expected_counts in expected_val_element_counts.items():
#             env_dict = val_partition.environments_by_element[el]
#             counts = sorted([len(x) for x in env_dict.values()])
#             assert counts == expected_counts

#         assert val == {'CCCCCCC', 'CC', 'CCCF', 'O', 'CCCO'}
