import pytest

from openff.nagl.label.dataset import LabelledDataset
from openff.nagl.label.labels import (
    LabelConformers,
    LabelCharges
)

pa = pytest.importorskip("pyarrow")
class TestLabelledDataset:

    def test_from_unmapped_smiles(self, small_dataset):
        assert small_dataset.dataset.count_rows() == 2
        assert small_dataset.dataset.schema.names == ["mapped_smiles"]

    def test_from_mapped_smiles(self, tmp_path):
        smiles = [
            '[C:1]([H:2])([H:3])([H:4])[H:5]',
            '[C:1]([C:2]([H:6])([H:7])[H:8])([H:3])([H:4])[H:5]'
        ]
        dataset = LabelledDataset.from_smiles(
            tmp_path,
            smiles,
            mapped=True
        )
        assert dataset.dataset.count_rows() == 2
        assert dataset.dataset.schema.names == ["mapped_smiles"]

        pydict = dataset.dataset.to_table().to_pydict()
        assert pydict["mapped_smiles"] == smiles

    def test_add_column(self, small_dataset):
        assert small_dataset.dataset.schema.names == ["mapped_smiles"]
        small_dataset._append_columns({"foo": [1, 2]})
        assert small_dataset.dataset.schema.names == ["mapped_smiles", "foo"]
        assert small_dataset.dataset.schema.field("foo").type == pa.int64()
    
    def test_apply_labellers(self, small_dataset):
        labellers = [
            LabelConformers(),
            LabelCharges(),
        ]
        small_dataset.apply_labellers(labellers)
        columns = ["mapped_smiles", "conformers", "n_conformers", "charges"]
        assert small_dataset.dataset.schema.names == columns
        
        table = small_dataset.dataset.to_table()
        pydict = table.to_pydict()
        assert pydict["n_conformers"] == [1, 1]
        assert isinstance(pydict["conformers"][0], list)

        expected_charges = [
            [0.0] * 5,
            [0.0] * 8
        ]
        assert pydict["charges"] == expected_charges