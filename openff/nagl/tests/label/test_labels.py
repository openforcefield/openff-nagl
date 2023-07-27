import pytest

from openff.nagl.label.labels import (
    LabelConformers,
    LabelCharges
)

class TestLabelCharges:

    def test_label_with_conformers_on_fly(self, small_dataset):
        labellers = [
            LabelConformers(),
            LabelCharges(use_existing_conformers=True),
        ]
        small_dataset.apply_labellers(labellers)
        columns = ["mapped_smiles", "conformers", "n_conformers", "charges"]
        assert small_dataset.dataset.schema.names == columns
        