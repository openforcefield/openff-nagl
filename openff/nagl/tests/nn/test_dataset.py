import os
import pathlib
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset
from openff.toolkit.topology.molecule import Molecule
from openff.units import unit

from openff.nagl.molecule._dgl import DGLMolecule, DGLMoleculeBatch
from openff.nagl.features.atoms import AtomConnectivity, AtomFormalCharge, AtomicElement
from openff.nagl.features.bonds import BondIsInRing, BondOrder
from openff.nagl.nn.dataset import (
    DGLMoleculeDatasetEntry,
    DGLMoleculeDataset,
)
from openff.nagl.tests.data.files import EXAMPLE_PARQUET_DATASET, EXAMPLE_FEATURIZED_PARQUET_DATASET
from openff.nagl.storage._store import (
    ConformerRecord,
    MoleculeRecord,
    MoleculeStore,
    PartialChargeRecord,
    WibergBondOrderRecord,
)

pytest.importorskip("dgl")

def label_formal_charge(molecule: Molecule):
    return {
        "formal_charges": torch.tensor(
            [
                atom.formal_charge.m_as(unit.elementary_charge)
                for atom in molecule.atoms
            ],
            dtype=torch.float,
        ),
    }


@pytest.fixture()
def example_pyarrow_table():
    columns = [
        'mapped_smiles', 'am1bcc_charges',
        'conformers', 'am1bcc_esps',
        'esp_lengths', 'am1bcc_dipoles'
    ]
    table = pq.read_table(
        EXAMPLE_FEATURIZED_PARQUET_DATASET,
        columns=columns
    )
    return table

@pytest.fixture()
def example_featurized_pyarrow_table():
    table = pq.read_table(EXAMPLE_FEATURIZED_PARQUET_DATASET)
    return table


@pytest.fixture()
def example_atom_features():
    return [AtomicElement(), AtomConnectivity()]

@pytest.fixture()
def example_bond_features():
    return [BondIsInRing()]


class TestDGLMoleculeDatasetEntry:

    def _assert_label_shapes(self, entry):
        assert isinstance(entry.labels, dict)
        assert len(entry.labels) == 5
        for value in entry.labels.values():
            assert isinstance(value, torch.Tensor)

        assert entry.labels["am1bcc_charges"].shape == (15,)
        assert entry.labels["conformers"].shape == (450,)
        assert entry.labels["am1bcc_esps"].shape == (8789,)
        assert entry.labels["am1bcc_dipoles"].shape == (30,)

        esp_lengths = entry.labels["esp_lengths"].detach().numpy()
        expected_lengths = [883, 885, 881, 882, 879, 884, 874, 875, 870, 876]
        assert np.allclose(esp_lengths, expected_lengths)

    def example_featurized_pyarrow_table(
        self,
        example_pyarrow_table,
        example_atom_features,
        example_bond_features,
    ):
        row = example_pyarrow_table.to_pylist()[1]
        entry = DGLMoleculeDatasetEntry._from_unfeaturized_pyarrow_row(
            row,
            atom_features=example_atom_features,
            bond_features=example_bond_features,
        )
        assert isinstance(entry, DGLMoleculeDatasetEntry)
        assert isinstance(entry.molecule, DGLMolecule)
        assert entry.molecule.n_atoms == 15
        assert entry.molecule.graph.ndata["feat"].shape == (15, 14)
        self._assert_label_shapes(entry)
    
    def test_from_featurized_row(
        self,
        example_featurized_pyarrow_table
    ):
        row = example_featurized_pyarrow_table.to_pylist()[1]
        entry = DGLMoleculeDatasetEntry._from_featurized_pyarrow_row(
            row,
            atom_feature_column="atom_features",
            bond_feature_column="bond_features",
        )
        assert isinstance(entry, DGLMoleculeDatasetEntry)
        assert isinstance(entry.molecule, DGLMolecule)
        assert entry.molecule.n_atoms == 15
        assert entry.molecule.graph.ndata["feat"].shape == (15, 14)
        self._assert_label_shapes(entry)
    

        
class TestDGLMoleculeDataset:

    @pytest.fixture()
    def featurized_dataset(self):
        return DGLMoleculeDataset.from_featurized_parquet(
            EXAMPLE_FEATURIZED_PARQUET_DATASET,
            atom_feature_column="atom_features",
            bond_feature_column="bond_features",
        )


    def test_from_unfeaturized_parquet(
        self,
        example_atom_features,
        example_bond_features,
    ):
        ds = DGLMoleculeDataset.from_unfeaturized_parquet(
            EXAMPLE_PARQUET_DATASET,
            atom_features=example_atom_features,
            bond_features=example_bond_features,
        )
        assert len(ds.entries) == 10
        for entry in ds.entries:
            assert entry.molecule.graph.ndata["feat"].shape[1] == 14
            assert len(entry.labels) == 5

    def test_from_featurized_parquet(self, featurized_dataset):
        assert len(featurized_dataset.entries) == 10
        for entry in featurized_dataset.entries:
            assert entry.molecule.graph.ndata["feat"].shape[1] == 14
            assert len(entry.labels) == 5

    def test_to_pyarrow(
        self,
        featurized_dataset,
        example_featurized_pyarrow_table,
    ):
        df = featurized_dataset.to_pyarrow().to_pandas()
        example = example_featurized_pyarrow_table.to_pandas()
        assert len(df.columns) == len(example.columns)

        for col in df.columns:
            if col == "mapped_smiles":
                assert np.array_equal(df[col].values, example[col].values)
            else:
                df_ = np.concatenate(df[col].values)
                example_ = np.concatenate(example[col].values)
                assert np.allclose(df_, example_)



    def test_from_openff(self, openff_methane_charged):
        data_set = DGLMoleculeDataset.from_openff(
            [openff_methane_charged],
            label_function=label_formal_charge,
            atom_features=[AtomConnectivity()],
            bond_features=[BondIsInRing()],
        )
        assert len(data_set) == 1
        assert data_set.n_atom_features == 4

        dgl_molecule, labels = data_set[0]
        assert isinstance(dgl_molecule, DGLMolecule)
        assert dgl_molecule.n_atoms == 5

        assert "formal_charges" in labels
        label = labels["formal_charges"]
        assert label.numpy().shape == (5,)

