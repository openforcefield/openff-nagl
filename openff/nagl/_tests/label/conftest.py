import pytest

from openff.nagl.label.dataset import LabelledDataset


@pytest.fixture(scope="function")
def small_dataset(tmp_path):
    smiles = ["C", "CC"]
    dataset = LabelledDataset.from_smiles(
        tmp_path,
        smiles,
        mapped=False
    )
    return dataset