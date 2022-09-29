import os
import pathlib
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset

from openff.nagl.features import (
    AtomFormalCharge,
    AtomicElement,
    BondOrder,
)
from openff.nagl.nn.data import DGLMoleculeDataset
from openff.nagl.nn.gcn.sage import SAGEConvStack
from openff.nagl.nn.modules.lightning import (
    ConvolutionModule,
    DGLMoleculeLightningDataModule,
    DGLMoleculeLightningModel,
    ReadoutModule,
)
from openff.nagl.nn.modules.pooling import (
    PoolAtomFeatures,
    PoolBondFeatures,
)
from openff.nagl.nn.modules.postprocess import ComputePartialCharges
from openff.nagl.nn.sequential import SequentialLayers
from openff.nagl.storage.record import (
    ChargeMethod,
    ConformerRecord,
    MoleculeRecord,
    PartialChargeRecord,
    WibergBondOrderMethod,
    WibergBondOrderRecord,
)
from openff.nagl.storage.store import MoleculeStore


@pytest.fixture()
def mock_atom_model() -> DGLMoleculeLightningModel:
    convolution = ConvolutionModule(
        n_input_features=4,
        hidden_feature_sizes=[4],
        architecture="SAGEConv",
    )
    readout_layers = SequentialLayers.with_layers(
        n_input_features=4,
        hidden_feature_sizes=[2],
    )
    model = DGLMoleculeLightningModel(
        convolution_module=convolution,
        readout_modules={
            "atom": ReadoutModule(
                pooling_layer=PoolAtomFeatures(),
                readout_layers=readout_layers,
                postprocess_layer=ComputePartialCharges(),
            ),
        },
        learning_rate=0.01,
    )
    return model


class TestDGLMoleculeLightningModel:
    def test_init(self):
        model = DGLMoleculeLightningModel(
            convolution_module=ConvolutionModule(
                n_input_features=1,
                hidden_feature_sizes=[2, 3],
                architecture="SAGEConv",
            ),
            readout_modules={
                "atom": ReadoutModule(
                    pooling_layer=PoolAtomFeatures(),
                    readout_layers=SequentialLayers.with_layers(
                        n_input_features=2,
                        hidden_feature_sizes=[2],
                        layer_activation_functions=["Identity"],
                    ),
                    postprocess_layer=ComputePartialCharges(),
                ),
                "bond": ReadoutModule(
                    pooling_layer=PoolBondFeatures(
                        layers=SequentialLayers.with_layers(
                            n_input_features=4,
                            hidden_feature_sizes=[4],
                        )
                    ),
                    readout_layers=SequentialLayers.with_layers(
                        n_input_features=4,
                        hidden_feature_sizes=[8],
                    ),
                ),
            },
            learning_rate=0.01,
        )

        assert model.convolution_module is not None
        assert isinstance(model.convolution_module, ConvolutionModule)

        assert isinstance(model.convolution_module.gcn_layers, SAGEConvStack)
        assert len(model.convolution_module.gcn_layers) == 2

        readouts = model.readout_modules
        assert all(x in readouts for x in ["atom", "bond"])

        assert isinstance(readouts["atom"].pooling_layer, PoolAtomFeatures)
        assert isinstance(readouts["bond"].pooling_layer, PoolBondFeatures)

        assert np.isclose(model.learning_rate, 0.01)

    def test_forward(self, mock_atom_model, dgl_methane):
        output = mock_atom_model.forward(dgl_methane)
        assert "atom" in output
        assert output["atom"].shape == (5, 1)

    @pytest.mark.parametrize(
        "method_name", ["training_step", "validation_step", "test_step"]
    )
    def test_step(self, mock_atom_model, method_name, dgl_methane, monkeypatch):
        def mock_forward(_):
            return {"atom": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])}

        monkeypatch.setattr(mock_atom_model, "forward", mock_forward)

        loss_function = getattr(mock_atom_model, method_name)
        fake_comparison = {"atom": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]])}
        loss = loss_function((dgl_methane, fake_comparison), 0)
        assert torch.isclose(loss, torch.tensor([1.0]))

    def test_configure_optimizers(self, mock_atom_model):
        optimizer = mock_atom_model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        assert torch.isclose(torch.tensor(
            optimizer.defaults["lr"]), torch.tensor(0.01))


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
            output_path="tmp.pkl",
            use_cached_data=True,
        )

    @pytest.fixture()
    def mock_data_store(self, tmpdir) -> str:
        store_path = os.path.join(tmpdir, "store.sqlite")
        conformer = ConformerRecord(
            coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            partial_charges=[PartialChargeRecord(
                method="am1bcc", values=[1.0, -1.0])],
            bond_orders=[WibergBondOrderRecord(
                method="am1", values=[(0, 1, 1.0)])],
        )

        store = MoleculeStore(store_path)
        store.store(
            MoleculeRecord(
                mapped_smiles="[Cl:1][Cl:2]",
                conformers=[conformer],
            )
        )

        return store_path

    @pytest.fixture()
    def mock_data_module_with_store(self, tmpdir, mock_data_store):
        data_module = DGLMoleculeLightningDataModule(
            atom_features=[AtomicElement(categories=["Cl", "H"])],
            bond_features=[BondOrder()],
            partial_charge_method="am1bcc",
            bond_order_method="am1",
            training_set_paths=mock_data_store,
            training_batch_size=None,
            validation_set_paths=mock_data_store,
            test_set_paths=mock_data_store,
            output_path=os.path.join(tmpdir, "tmp.pkl"),
        )
        return data_module

    def test_init(self, mock_data_module):
        assert isinstance(mock_data_module.atom_features[0], AtomicElement)
        assert mock_data_module.n_atom_features == 5

        assert isinstance(mock_data_module.bond_features[0], BondOrder)

        assert mock_data_module.partial_charge_method == "am1bcc"
        assert mock_data_module.bond_order_method == "am1"

        assert mock_data_module.training_set_paths == [
            pathlib.Path("train.sqlite")]
        assert mock_data_module.training_batch_size == 1

        assert mock_data_module.validation_set_paths == [
            pathlib.Path("val.sqlite")]
        assert mock_data_module.validation_batch_size == 2

        assert mock_data_module.test_set_paths == [pathlib.Path("test.sqlite")]
        assert mock_data_module.test_batch_size == 3

        assert mock_data_module.output_path == pathlib.Path("tmp.pkl")
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
        dataset = mock_data_module._prepare_data_from_paths(
            [mock_data_store] * 2)
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset.datasets) == 2
        assert len(dataset) == 2

    def test_prepare_data_from_path_error(self, mock_data_module):
        with pytest.raises(NotImplementedError, match="Only paths to SQLite"):
            mock_data_module._prepare_data_from_paths("tmp.pkl")

    def test_prepare(self, mock_data_module_with_store):
        mock_data_module_with_store.prepare_data()

        assert os.path.isfile(mock_data_module_with_store.output_path)
        with open(mock_data_module_with_store.output_path, "rb") as file:
            datasets = pickle.load(file)

        assert all(isinstance(dataset, ConcatDataset) for dataset in datasets)
        assert all(dataset.datasets[0].n_features == 2 for dataset in datasets)

    def test_prepare_cache(self, mock_data_module_with_store, monkeypatch):
        with open(mock_data_module_with_store.output_path, "wb") as file:
            pickle.dump((None, None, None), file)
        monkeypatch.setattr(mock_data_module_with_store,
                            "use_cached_data", True)
        mock_data_module_with_store.prepare_data()

    def test_error_on_cache(self, mock_data_module_with_store, monkeypatch):
        with open(mock_data_module_with_store.output_path, "wb") as file:
            pickle.dump((None, None, None), file)
        monkeypatch.setattr(mock_data_module_with_store,
                            "use_cached_data", False)

        with pytest.raises(FileExistsError):
            mock_data_module_with_store.prepare_data()

    def test_setup(self, mock_data_module_with_store, monkeypatch):
        monkeypatch.setattr(mock_data_module_with_store, "test_set_paths", [])
        mock_data_module_with_store.prepare_data()
        mock_data_module_with_store.setup()

        assert isinstance(
            mock_data_module_with_store._train_data.datasets[0], DGLMoleculeDataset
        )
        assert isinstance(
            mock_data_module_with_store._val_data.datasets[0], DGLMoleculeDataset
        )
        assert mock_data_module_with_store._test_data is None
