import itertools
import pathlib
import pytest
import shutil

import torch
import numpy as np
import pytorch_lightning as pl

from openff.nagl.training.training import DGLMoleculeDataModule, DataHash, TrainingGNNModel
from openff.nagl.nn._models import GNNModel
from openff.nagl.nn._dataset import (
    DGLMoleculeDataLoader,
    DGLMoleculeDataset,
    _LazyDGLMoleculeDataset
)
from openff.nagl.config.training import TrainingConfig
from openff.nagl.tests.data.files import (
    EXAMPLE_UNFEATURIZED_PARQUET_DATASET,
    EXAMPLE_FEATURIZED_PARQUET_DATASET,
    EXAMPLE_UNFEATURIZED_PARQUET_DATASET_SHORT,
    EXAMPLE_TRAINING_CONFIG,
    EXAMPLE_TRAINING_CONFIG_LAZY,
    EXAMPLE_FEATURIZED_LAZY_DATA,
    EXAMPLE_FEATURIZED_LAZY_DATA_SHORT,
)

dgl = pytest.importorskip("dgl")

@pytest.fixture
def example_training_config():
    config = TrainingConfig.from_yaml(EXAMPLE_TRAINING_CONFIG)
    return config


class TestDataHash:
    def test_to_hash(self, example_atom_features, example_bond_features):
        all_filenames = [EXAMPLE_UNFEATURIZED_PARQUET_DATASET, EXAMPLE_FEATURIZED_PARQUET_DATASET]
        all_atom_features = [example_atom_features, []]
        all_bond_features = [example_bond_features, []]
        all_columns = [["a", "b"], ["c"], []]

        all_combinations = list(itertools.product(
            all_filenames, all_columns, all_atom_features, all_bond_features
        ))
        all_hashers = []
        for fn, cols, atom_features, bond_features in all_combinations:
            hasher = DataHash.from_file(
                fn,
                columns=cols,
                atom_features=atom_features,
                bond_features=bond_features,
            )
            all_hashers.append(hasher)
        for hasher, combination in zip(all_hashers, all_combinations):
            fn, cols, atom_features, bond_features = combination
            hasher2 = DataHash.from_file(
                fn,
                columns=cols,
                atom_features=atom_features,
                bond_features=bond_features,
            )
            assert hasher.to_hash() == hasher2.to_hash()

        while all_hashers:
            hasher = all_hashers.pop(0)
            for hasher2 in all_hashers:
                assert hasher.to_hash() != hasher2.to_hash()


class TestDGLMoleculeDataModule:
    def test_init(self, example_training_config):
        data_module = DGLMoleculeDataModule(example_training_config)
        for stage in ["train", "val", "test"]:
            assert stage in data_module._dataset_configs
        assert data_module.config is example_training_config
        assert isinstance(data_module.train_dataloader(), DGLMoleculeDataLoader)
        assert data_module.train_dataloader().batch_size == 5
        assert isinstance(data_module.val_dataloader(), DGLMoleculeDataLoader)
        assert data_module.val_dataloader().batch_size == 2
        assert data_module.test_dataloader is None

    @pytest.mark.parametrize(
        "filename, hash_value",
        [
            (EXAMPLE_UNFEATURIZED_PARQUET_DATASET, "9e89f05d67df7ba8efbfd7d27eea31b436218fb5f0387b24dfa0cc9552c764ea"),
            (EXAMPLE_UNFEATURIZED_PARQUET_DATASET_SHORT, "95da5126cc02a66d5f34388ac2aa735046622ba7b248c67168c3ae37a287321d"),
        ]
    )
    def test_hash_file(self, example_training_config, filename, hash_value):
        data_module = DGLMoleculeDataModule(example_training_config)
        file_hash = data_module._get_hash_file([filename], ["a", "b"])
        assert file_hash == pathlib.Path(hash_value)

    def test_setup(self, tmpdir, example_training_config):
        data_module = DGLMoleculeDataModule(example_training_config)
        assert len(data_module._datasets) == 0

        with tmpdir.as_cwd():
            shutil.copytree(
                EXAMPLE_UNFEATURIZED_PARQUET_DATASET.resolve(),
                EXAMPLE_UNFEATURIZED_PARQUET_DATASET.stem
            )
            shutil.copytree(
                EXAMPLE_UNFEATURIZED_PARQUET_DATASET_SHORT.resolve(),
                EXAMPLE_UNFEATURIZED_PARQUET_DATASET_SHORT.stem
            )
            for stage in ["train", "val", "test"]:
                config = data_module._dataset_configs[stage]
                config = config.copy(
                    update={
                        "use_cached_data": True,
                        "cache_directory": ".",
                    }
                )
                data_module._dataset_configs[stage] = config

            data_module.prepare_data()
            assert len(data_module._datasets) == 0

            data_module.setup()
            training_set = data_module._datasets["train"]
            assert len(training_set) == 14
            assert len(training_set.datasets) == 2
            assert isinstance(training_set.datasets[0], DGLMoleculeDataset)
            assert isinstance(training_set.datasets[1], DGLMoleculeDataset)
            assert training_set.datasets[0].n_atom_features == 25

            validation_set = data_module._datasets["val"]
            assert len(validation_set) == 4
            assert len(validation_set.datasets) == 1
            assert isinstance(validation_set.datasets[0], DGLMoleculeDataset)
            assert validation_set.datasets[0].n_atom_features == 25
            assert data_module._datasets["test"] is None

            train_dataloader = data_module.train_dataloader()
            assert train_dataloader.batch_size == 5
            assert len(train_dataloader) == 3  # 3 batches of 5 (total 14)
            val_dataloader = data_module.val_dataloader()
            assert val_dataloader.batch_size == 2
            assert len(val_dataloader) == 2  # 2 batches of 2 (total 4)

    def test_setup_lazy(self, tmpdir):
        example_training_config = TrainingConfig.from_yaml(
            EXAMPLE_TRAINING_CONFIG_LAZY
        )
        data_module = DGLMoleculeDataModule(example_training_config)
        assert len(data_module._datasets) == 0

        with tmpdir.as_cwd():
            shutil.copy(
                EXAMPLE_FEATURIZED_LAZY_DATA.resolve(),
                "."
            )
            shutil.copy(
                EXAMPLE_FEATURIZED_LAZY_DATA_SHORT.resolve(),
                "."
            )
            shutil.copytree(
                EXAMPLE_UNFEATURIZED_PARQUET_DATASET.resolve(),
                EXAMPLE_UNFEATURIZED_PARQUET_DATASET.stem
            )
            shutil.copytree(
                EXAMPLE_UNFEATURIZED_PARQUET_DATASET_SHORT.resolve(),
                EXAMPLE_UNFEATURIZED_PARQUET_DATASET_SHORT.stem
            )
            for stage in ["train", "val", "test"]:
                config = data_module._dataset_configs[stage]
                config = config.copy(
                    update={
                        "use_cached_data": True,
                        "cache_directory": ".",
                    }
                )
                data_module._dataset_configs[stage] = config

            data_module.prepare_data()
            assert len(data_module._datasets) == 0

            data_module.setup()
            training_set = data_module._datasets["train"]
            assert len(training_set) == 14
            assert len(training_set.datasets) == 2
            assert isinstance(training_set.datasets[0], _LazyDGLMoleculeDataset)
            assert isinstance(training_set.datasets[1], _LazyDGLMoleculeDataset)
            assert training_set.datasets[0].n_atom_features == 25

            validation_set = data_module._datasets["val"]
            assert len(validation_set) == 4
            assert len(validation_set.datasets) == 1
            assert isinstance(validation_set.datasets[0], DGLMoleculeDataset)
            assert validation_set.datasets[0].n_atom_features == 25
            assert data_module._datasets["test"] is None

            train_dataloader = data_module.train_dataloader()
            assert train_dataloader.batch_size == 5
            assert len(train_dataloader) == 3  # 3 batches of 5 (total 14)
            val_dataloader = data_module.val_dataloader()
            assert val_dataloader.batch_size == 2
            assert len(val_dataloader) == 2  # 2 batches of 2 (total 4)






    # def test_prepare_data_uncached(self, tmpdir, example_training_config):
    #     data_module = DGLMoleculeDataModule(example_training_config)
    #     assert len(data_module._datasets) == 0

    #     with tmpdir.as_cwd():
    #         shutil.copytree(
    #             EXAMPLE_UNFEATURIZED_PARQUET_DATASET.resolve(),
    #             EXAMPLE_UNFEATURIZED_PARQUET_DATASET.stem
    #         )
    #         shutil.copytree(
    #             EXAMPLE_UNFEATURIZED_PARQUET_DATASET_SHORT.resolve(),
    #             EXAMPLE_UNFEATURIZED_PARQUET_DATASET_SHORT.stem
    #         )
    #         for stage in ["train", "val", "test"]:
    #             config = data_module._dataset_configs[stage]
    #             config = config.copy(
    #                 update={
    #                     "use_cached_data": False,
    #                     "cache_directory": ".",
    #                 }
    #             )
    #             data_module._dataset_configs[stage] = config

    #         data_module.prepare_data()
    #         assert len(data_module._datasets) == 0

    #         data_module.setup()
    #         assert len(data_module._datasets) == 3
    #         assert len(data_module._datasets["train"].entries) == 14
    #         assert data_module._datasets["train"].n_atom_features == 25
    #         assert len(data_module._datasets["val"].entries) == 4
    #         assert data_module._datasets["val"].n_atom_features == 25
    #         assert data_module._datasets["test"] is None





class TestTrainingGNNModel:
    @pytest.fixture()
    def example_training_model(self, example_training_config):
        model = TrainingGNNModel(example_training_config)
        return model
    
    @pytest.fixture()
    def mock_training_model(self, example_training_model, monkeypatch):
        def mock_forward(_):
            return {"am1bcc_charges": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])}
        monkeypatch.setattr(example_training_model, "forward", mock_forward)
        return example_training_model

    def test_roundtrip(self, example_training_model, tmpdir):
        with tmpdir.as_cwd():
            yaml_file = "model.yaml"
            example_training_model.config.to_yaml(yaml_file)
            model = TrainingGNNModel.from_yaml(yaml_file)
            assert model.config == example_training_model.config
    
    def test_init(self, example_training_model, example_training_config):
        assert example_training_model.config == example_training_config
        assert example_training_model.hparams["config"] == example_training_config
        assert isinstance(example_training_model.model, GNNModel)

    def test_configure_optimizers(self, example_training_model):
        optimizer = example_training_model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        assert torch.isclose(torch.tensor(optimizer.defaults["lr"]), torch.tensor(0.001))


    def test_unweighted_readout_test_step(self, mock_training_model, dgl_methane):
        labels = {
            "am1bcc_charges": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),
        }

        loss = mock_training_model.test_step((dgl_methane, labels), 0)
        assert torch.isclose(loss["test_loss"], torch.tensor([1.0]))

    def test_weighted_readout_validation_step(self, mock_training_model, dgl_methane):
        labels = {
            "am1bcc_charges": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),
        }

        loss = mock_training_model.validation_step((dgl_methane, labels), 0)
        assert torch.isclose(loss["validation_loss"], torch.tensor([50.0]))

    def test_weighted_mixed_training_step(self, mock_training_model, dgl_methane):
        fake_conformers = torch.tensor(np.arange(30)).float()
        n_conformers = torch.tensor([2,])
        inv_dist = torch.tensor(
            [
                0.05773503, 0.08247861, 0.14433757, 0.57735027, 0.28867513,
                0.04441156, 0.05773503, 0.08247861, 0.14433757, 0.57735027,
                0.03608439, 0.04441156, 0.05773503, 0.08247861, 0.14433757,
                0.14433757, 0.57735027, 0.28867513, 0.11547005, 0.07216878,
                0.08247861, 0.14433757, 0.57735027, 0.28867513, 0.11547005,
            ]
       )
        reference_esps = np.array(
            [5.55905831, 4.77773209, 1.71476202, 4.18578945, 5.04356699]
        )
        reference_dipoles = np.array([150., 170., 190., 450., 470., 490.])

        labels = {
            "am1bcc_charges": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),

            # esp
            "esp_grid_inverse_distances": inv_dist,
            "am1bcc_esps": torch.tensor(reference_esps),
            "esp_lengths": torch.tensor([3, 2,]),

            # multi_dipole
            "conformers": fake_conformers,
            "n_conformers": n_conformers,
            "am1bcc_dipoles": torch.tensor(reference_dipoles),

        }

        expected_dipoles = np.array([120., 135., 150., 345., 360., 375.])
        expected_esps = np.array(
            [4.4084817 , 3.87141906, 1.34971487, 2.98778764, 3.83525536]
        )

        mse_esps = ((reference_esps - expected_esps) ** 2).mean()
        mae_dipoles = (np.abs(reference_dipoles - expected_dipoles)).mean()
        rmse_readout = 50.0
        expected_loss = mse_esps + mae_dipoles + rmse_readout

        loss = mock_training_model.training_step((dgl_methane, labels), 0)
        loss = loss["loss"]
        assert torch.isclose(loss, torch.tensor([expected_loss], dtype=torch.float32))
        assert torch.isclose(loss, torch.tensor([123.534743]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_model_no_error(example_training_config):
    model = TrainingGNNModel(example_training_config)
    data = DGLMoleculeDataModule(example_training_config)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=2)
    trainer.fit(model, datamodule=data)
