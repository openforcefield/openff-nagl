import os
import pathlib
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset
from openff.toolkit.topology.molecule import Molecule
from openff.units import unit

from openff.nagl.molecule._dgl import DGLMolecule, DGLMoleculeBatch
from openff.nagl.features.atoms import AtomConnectivity, AtomFormalCharge, AtomicElement
from openff.nagl.features.bonds import BondIsInRing, BondOrder
from openff.nagl.nn._dataset import (
    DGLMoleculeDatasetEntry,
    DGLMoleculeDataset,
    _LazyDGLMoleculeDataset,
    DGLMoleculeDataLoader,
    DataHash,
    _get_hashed_arrow_dataset_path
)
from openff.nagl.tests.data.files import EXAMPLE_UNFEATURIZED_PARQUET_DATASET, EXAMPLE_FEATURIZED_PARQUET_DATASET

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


class TestDataHash:

    def test_hash_empty(self):
        hasher = DataHash(
            path_hash="path hash",
            columns=["multiple", "columns"],
            atom_features=[AtomConnectivity()],
            bond_features=[BondIsInRing()],
        )
        hash_value = hasher.to_hash()
        assert hash_value == "0c25874901b9b5fe2e16434749c9aef01ff4d53c7f04d2318052d77a70ad98bc"

    def test_from_file(self):
        hasher = DataHash.from_file(
            "/path/to/file.parquet",
            columns=["multiple", "columns"],
            atom_features=None,
            bond_features=None,
        )
        hash_value = hasher.to_hash()
        assert hash_value == "ce6af226f485d344156d135a51e2ce79282a457a78565999574224bb6469cbf0"


def test_get_hashed_arrow_dataset_path():
    path = _get_hashed_arrow_dataset_path(
        "/path/to/file.parquet",
        columns=["multiple", "columns"],
        atom_features=None,
        bond_features=None,
        directory="test"
    )
    expected_path = pathlib.Path("test") / "ce6af226f485d344156d135a51e2ce79282a457a78565999574224bb6469cbf0"
    assert path == expected_path


# @pytest.fixture()
# def example_pyarrow_table():
#     columns = [
#         'mapped_smiles', 'am1bcc_charges',
#         'conformers', 'am1bcc_esps',
#         'esp_lengths', 'am1bcc_dipoles'
#     ]
#     table = pq.read_table(
#         EXAMPLE_FEATURIZED_PARQUET_DATASET,
#         columns=columns
#     )
#     return table

# @pytest.fixture()
# def example_featurized_pyarrow_table():
#     table = pq.read_table(EXAMPLE_FEATURIZED_PARQUET_DATASET)
#     return table


@pytest.fixture()
def featurized_dataset():
    return DGLMoleculeDataset.from_arrow_dataset(
        EXAMPLE_FEATURIZED_PARQUET_DATASET,
        atom_feature_column="atom_features",
        bond_feature_column="bond_features",
    )


class TestDGLMoleculeDatasetEntry:

    def test_from_openff(self, openff_methyl_methanoate):
        entry = DGLMoleculeDatasetEntry.from_openff(
            openff_methyl_methanoate,
            labels={"label": np.zeros((8, 1))},
            atom_features=[AtomConnectivity()],
            bond_features=None,
        )
        assert len(entry.labels) == 1
        assert entry.labels["label"].shape == (8, 1)

    def _assert_label_shapes(self, entry):
        assert isinstance(entry.labels, dict)
        assert len(entry.labels) == 7
        for value in entry.labels.values():
            assert isinstance(value, torch.Tensor)

        assert entry.labels["am1bcc_charges"].shape == (15,)
        assert entry.labels["conformers"].shape == (405,)
        assert entry.labels["am1bcc_esps"].shape == (7913,)
        assert entry.labels["am1bcc_dipoles"].shape == (27,)
        # assert entry.labels["n_conformers"].shape == (1,)
        assert entry.labels["esp_grid_inverse_distances"].shape == (7913 * 15,)

        esp_lengths = entry.labels["esp_lengths"].detach().numpy()
        expected_lengths = [883, 885, 881, 882, 879, 884, 874, 875, 870]
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
    
    # def test_from_featurized_row(
    #     self,
    #     example_featurized_pyarrow_table
    # ):
    #     row = example_featurized_pyarrow_table.to_pylist()[1]
    #     entry = DGLMoleculeDatasetEntry._from_featurized_pyarrow_row(
    #         row,
    #         atom_feature_column="atom_features",
    #         bond_feature_column="bond_features",
    #     )
    #     assert isinstance(entry, DGLMoleculeDatasetEntry)
    #     assert isinstance(entry.molecule, DGLMolecule)
    #     assert entry.molecule.n_atoms == 15
    #     assert entry.molecule.graph.ndata["feat"].shape == (15, 14)
    #     self._assert_label_shapes(entry)
    

        
class TestDGLMoleculeDataset:

    def test_from_featurized_parquet(self, featurized_dataset):
        assert len(featurized_dataset) == 10
        for entry in featurized_dataset.entries:
            assert entry.molecule.graph.ndata["feat"].shape[1] == 14
            assert len(entry.labels) == 7

    def test_from_arrow_dataset(
        self,
        example_atom_features,
        example_bond_features,
    ):
        ds = DGLMoleculeDataset.from_arrow_dataset(
            EXAMPLE_UNFEATURIZED_PARQUET_DATASET,
            atom_features=example_atom_features,
            bond_features=example_bond_features,
        )

        assert len(ds) == 10
        expected = {
            "am1bcc_charges", "conformers", "am1bcc_esps",
            "esp_lengths", "am1bcc_dipoles", "n_conformers",
            "esp_grid_inverse_distances",
        }
        for entry in ds:
            assert entry.molecule.graph.ndata["feat"].shape[1] == 14
            assert len(entry.labels) == 7
            assert set(entry.labels.keys()) == expected


    def test_to_pyarrow(
        self,
        featurized_dataset,
    ):
        df = featurized_dataset.to_pyarrow().to_pandas()
        example = ds.dataset(EXAMPLE_FEATURIZED_PARQUET_DATASET).to_table().to_pandas()

        assert len(df.columns) == len(example.columns)

        for col in df.columns:
            if col == "mapped_smiles":
                # can get serialized into different orders
                pass
            elif col == "n_conformers":
                assert np.array_equal(df[col].values, example[col].values)
            else:
                df_ = np.concatenate(df[col].values)
                example_ = np.concatenate(example[col].values)
                assert np.allclose(df_, example_)

    def test_from_openff(self, openff_methane_charged):
        data_set = DGLMoleculeDataset.from_openff(
            [openff_methane_charged],
            label_function=label_formal_charge,
            atom_features=[AtomConnectivity(categories=[1, 4])],
            bond_features=[BondIsInRing()],
        )
        assert len(data_set) == 1
        assert data_set.n_atom_features == 2

        dgl_molecule, labels = data_set[0]
        assert isinstance(dgl_molecule, DGLMolecule)
        assert dgl_molecule.n_atoms == 5

        assert "formal_charges" in labels
        label = labels["formal_charges"]
        assert label.numpy().shape == (5,)



class TestLazyDGLMoleculeDataset:

    def test_from_featurized_parquet(self, tmpdir):
        with tmpdir.as_cwd():
            featurized_dataset = _LazyDGLMoleculeDataset.from_arrow_dataset(
                EXAMPLE_FEATURIZED_PARQUET_DATASET,
                atom_feature_column="atom_features",
                bond_feature_column="bond_features",
            )
            assert len(featurized_dataset) == 10
            for entry in featurized_dataset:
                assert entry.molecule.graph.ndata["feat"].shape[1] == 14
                assert len(entry.labels) == 7

    def test_from_arrow_dataset(
        self,
        example_atom_features,
        example_bond_features,
        tmpdir
    ):
        with tmpdir.as_cwd():
            ds = _LazyDGLMoleculeDataset.from_arrow_dataset(
                EXAMPLE_UNFEATURIZED_PARQUET_DATASET,
                atom_features=example_atom_features,
                bond_features=example_bond_features,
            )

            assert len(ds) == 10
            expected = {
                "am1bcc_charges", "conformers", "am1bcc_esps",
                "esp_lengths", "am1bcc_dipoles", "n_conformers",
                "esp_grid_inverse_distances",
            }
            for entry in ds:
                assert entry.molecule.graph.ndata["feat"].shape[1] == 14
                assert len(entry.labels) == 7
                assert set(entry.labels.keys()) == expected


class TestDGLMoleculeDataLoader:
    def test_init(self, featurized_dataset):
        loader = DGLMoleculeDataLoader(featurized_dataset)
        entries = [*loader]
        assert len(entries) == 10

        for dgl_molecule, labels in entries:
            assert isinstance(dgl_molecule, DGLMoleculeBatch)
            assert len(labels) == 7


