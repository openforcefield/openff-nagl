import os
import pathlib
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset
from openff.toolkit.topology.molecule import Molecule
from openff.toolkit.topology.molecule import unit as off_unit

from openff.nagl._dgl import DGLMolecule, DGLMoleculeBatch
from openff.nagl.features.atoms import AtomConnectivity, AtomFormalCharge, AtomicElement
from openff.nagl.features.bonds import BondIsInRing, BondOrder
from openff.nagl.nn.dataset import DGLMoleculeDataLoader, DGLMoleculeDataset, DGLMoleculeLightningDataModule
from openff.nagl.storage._store import (
    ConformerRecord,
    MoleculeRecord,
    MoleculeStore,
    PartialChargeRecord,
    WibergBondOrderRecord,
)


def label_formal_charge(molecule: Molecule):
    return {
        "formal_charges": torch.tensor(
            [
                float(atom.formal_charge / off_unit.elementary_charge)
                for atom in molecule.atoms
            ],
            dtype=torch.float,
        ),
    }


def test_data_set_from_molecules(openff_methane_charged):

    data_set = DGLMoleculeDataset.from_openff(
        [openff_methane_charged],
        label_function=label_formal_charge,
        atom_features=[AtomConnectivity()],
        bond_features=[BondIsInRing()],
    )
    assert len(data_set) == 1
    assert data_set.n_features == 4

    dgl_molecule, labels = data_set[0]
    assert isinstance(dgl_molecule, DGLMolecule)
    assert dgl_molecule.n_atoms == 5

    assert "formal_charges" in labels
    label = labels["formal_charges"]
    assert label.numpy().shape == (5,)


def test_data_set_from_molecule_stores(tmpdir):

    charges = PartialChargeRecord(method="am1", values=[0.1, -0.1])
    bond_orders = WibergBondOrderRecord(
        method="am1",
        values=[(0, 1, 1.1)],
    )
    molecule_record = MoleculeRecord(
        mapped_smiles="[Cl:1]-[H:2]",
        conformers=[
            ConformerRecord(
                coordinates=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                partial_charges=[charges],
                bond_orders=[bond_orders],
            )
        ],
    )

    molecule_store = MoleculeStore(os.path.join(tmpdir, "store.sqlite"))
    molecule_store.store(records=[molecule_record])

    data_set = DGLMoleculeDataset.from_molecule_stores(
        molecule_stores=[molecule_store],
        atom_features=[AtomConnectivity()],
        bond_features=[BondIsInRing()],
        partial_charge_method="am1",
        bond_order_method="am1",
    )

    assert len(data_set) == 1
    assert data_set.n_features == 4

    dgl_molecule, labels = data_set[0]

    assert isinstance(dgl_molecule, DGLMolecule)
    assert dgl_molecule.n_atoms == 2
    assert "am1-charges" in labels
    assert labels["am1-charges"].numpy().shape == (2,)
    assert np.allclose(labels["am1-charges"].numpy(), [0.1, -0.1])
    assert "am1-wbo" in labels
    assert labels["am1-wbo"].numpy().shape == (1,)
    assert np.allclose(labels["am1-wbo"].numpy(), [1.1])


def test_data_set_loader():
    data_loader = DGLMoleculeDataLoader(
        dataset=DGLMoleculeDataset.from_openff(
            molecules=[Molecule.from_smiles("C"), Molecule.from_smiles("C[O-]")],
            label_function=label_formal_charge,
            atom_features=[AtomConnectivity()],
        ),
    )

    entries = [*data_loader]
    for dgl_molecule, labels in entries:
        assert isinstance(
            dgl_molecule, DGLMoleculeBatch
        ) and dgl_molecule.n_atoms_per_molecule == (5,)
        assert "formal_charges" in labels



class TestDGLMoleculeLightningDataModule:
    @pytest.fixture()
    def mock_data_module(self) -> DGLMoleculeLightningDataModule:
        atom_features = [
            AtomicElement(categories=["C", "H", "Cl"]),
            AtomFormalCharge(categories=[0, 1]),
        ]
        return DGLMoleculeLightningDataModule(
            atom_features=atom_features,
            bond_features=[BondOrder()],
            partial_charge_method="am1bcc",
            bond_order_method="am1",
            training_set_paths="train.sqlite",
            training_batch_size=1,
            validation_set_paths="val.sqlite",
            validation_batch_size=2,
            test_set_paths="test.sqlite",
            test_batch_size=3,
            data_cache_directory="tmp",
            use_cached_data=True,
        )

    @pytest.fixture()
    def mock_data_store(self, tmpdir) -> str:
        store_path = os.path.join(tmpdir, "store.sqlite")
        conformer = ConformerRecord(
            coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            partial_charges=[PartialChargeRecord(method="am1bcc", values=[1.0, -1.0])],
            bond_orders=[WibergBondOrderRecord(method="am1", values=[(0, 1, 1.0)])],
        )

        store = MoleculeStore(store_path)
        store.store(
            MoleculeRecord(
                mapped_smiles="[Cl:1][Cl:2]",
                conformers=[conformer],
            )
        )

        return store_path

    def create_mock_data_module(
        self, tmpdir, mock_data_store, use_cached_data: bool = True, test_set_paths=None
    ):
        if test_set_paths is None:
            test_set_paths = mock_data_store
        data_module = DGLMoleculeLightningDataModule(
            atom_features=[AtomicElement(categories=["Cl", "H"])],
            bond_features=[BondOrder()],
            partial_charge_method="am1bcc",
            bond_order_method="am1",
            training_set_paths=mock_data_store,
            training_batch_size=None,
            validation_set_paths=mock_data_store,
            test_set_paths=test_set_paths,
            data_cache_directory=os.path.join(tmpdir, "tmp"),
            use_cached_data=use_cached_data,
        )
        return data_module

    @pytest.fixture(scope="function")
    def mock_data_module_with_store(self, tmpdir, mock_data_store):
        return self.create_mock_data_module(tmpdir, mock_data_store)

    def test_without_cache(self, tmpdir, mock_data_store):
        data_module = self.create_mock_data_module(
            tmpdir=tmpdir, mock_data_store=mock_data_store, use_cached_data=False
        )

    def test_init(self, mock_data_module):
        assert isinstance(mock_data_module.atom_features[0], AtomicElement)
        assert mock_data_module.n_atom_features == 5

        assert isinstance(mock_data_module.bond_features[0], BondOrder)

        assert mock_data_module.partial_charge_method == "am1bcc"
        assert mock_data_module.bond_order_method == "am1"

        assert mock_data_module.training_set_paths == [pathlib.Path("train.sqlite")]
        assert mock_data_module.training_batch_size == 1

        assert mock_data_module.validation_set_paths == [pathlib.Path("val.sqlite")]
        assert mock_data_module.validation_batch_size == 2

        assert mock_data_module.test_set_paths == [pathlib.Path("test.sqlite")]
        assert mock_data_module.test_batch_size == 3

        assert mock_data_module.data_cache_directory == pathlib.Path("tmp")
        assert mock_data_module.use_cached_data is True

    def test__prepare_data_from_paths(self, mock_data_module, mock_data_store):
        dataset = mock_data_module._prepare_data_from_paths([mock_data_store])
        assert isinstance(dataset, ConcatDataset)

        dataset = dataset.datasets[0]
        assert isinstance(dataset, DGLMoleculeDataset)

        assert dataset.n_features == 5
        assert len(dataset) == 1

        molecule, labels = next(iter(dataset))

        assert molecule.n_atoms == 2
        assert molecule.n_bonds == 1
        assert {*labels} == {"am1bcc-charges", "am1-wbo"}

    def test_prepare_data_from_multiple_paths(self, mock_data_module, mock_data_store):
        dataset = mock_data_module._prepare_data_from_paths([mock_data_store] * 2)
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset.datasets) == 2
        assert len(dataset) == 2

    def test_prepare_data_from_path_error(self, mock_data_module):
        with pytest.raises(NotImplementedError, match="Only paths to SQLite"):
            mock_data_module._prepare_data_from_paths("tmp.pkl")

    def test_prepare(self, mock_data_module_with_store):
        mock_data_module_with_store.prepare_data()

        assert os.path.isfile(mock_data_module_with_store._training_cache_path)
        with open(mock_data_module_with_store._training_cache_path, "rb") as file:
            dataset = pickle.load(file)

        assert isinstance(dataset, ConcatDataset)
        assert isinstance(dataset.datasets[0], DGLMoleculeDataset)
        assert dataset.datasets[0].n_features == 2

    def test_prepare_cache(self, tmpdir, mock_data_store):
        mock_data_module_with_store = self.create_mock_data_module(
            tmpdir, mock_data_store, use_cached_data=True
        )
        mock_data_module_with_store.data_cache_directory.mkdir(
            exist_ok=True, parents=True
        )
        with open(mock_data_module_with_store._training_cache_path, "wb") as file:
            pickle.dump("test", file)

        assert (
            mock_data_module_with_store._training_cache_path
            == mock_data_module_with_store._validation_cache_path
        )
        assert (
            mock_data_module_with_store._training_cache_path
            == mock_data_module_with_store._test_cache_path
        )

        mock_data_module_with_store.prepare_data()
        mock_data_module_with_store.setup()

        # all paths will be the same file, since datasets are the same
        assert mock_data_module_with_store._train_data == "test"
        assert mock_data_module_with_store._val_data == "test"
        assert mock_data_module_with_store._test_data == "test"

    def test_error_on_cache(self, tmpdir, mock_data_store):
        mock_data_module_with_store = self.create_mock_data_module(
            tmpdir, mock_data_store, use_cached_data=False
        )
        mock_data_module_with_store.data_cache_directory.mkdir(
            exist_ok=True, parents=True
        )
        with open(mock_data_module_with_store._training_cache_path, "wb") as file:
            pickle.dump("test", file)

        with pytest.raises(FileExistsError):
            mock_data_module_with_store.prepare_data()

    def test_setup(self, tmpdir, mock_data_store):
        mock_data_module_with_store = self.create_mock_data_module(
            tmpdir, mock_data_store, use_cached_data=True, test_set_paths=[]
        )
        mock_data_module_with_store.prepare_data()
        mock_data_module_with_store.setup()

        assert isinstance(
            mock_data_module_with_store._train_data.datasets[0], DGLMoleculeDataset
        )
        assert isinstance(
            mock_data_module_with_store._val_data.datasets[0], DGLMoleculeDataset
        )
        assert mock_data_module_with_store._test_data is None
