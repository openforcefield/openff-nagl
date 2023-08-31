import numpy as np
import torch

from openff.nagl.training.loss import (
    HeavyAtomReadoutTarget,
    MultipleDipoleTarget,
    MultipleESPTarget,
    ReadoutTarget,
    SingleDipoleTarget,
)


class TestReadoutTarget:
    def test_single_molecule(self, dgl_methane):
        predictions = {
            "am1bcc_charges": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        }
        labels = {
            "am1bcc_charges": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),
        }

        target = ReadoutTarget(
            metric="rmse",
            target_label="am1bcc_charges",
            prediction_label="am1bcc_charges",
        )
        assert target.get_required_columns() == ["am1bcc_charges"]

        loss = target.evaluate_loss(dgl_methane, labels, predictions, {})
        assert torch.isclose(loss, torch.tensor([1.0]))

    def test_batch(self, dgl_batch):
        predictions = {"am1bcc_charges1": torch.tensor(np.arange(28)).float()}
        labels = {"am1bcc_charges2": torch.tensor(np.arange(28) / 2).float()}

        target = ReadoutTarget(
            metric="mae",
            target_label="am1bcc_charges2",
            prediction_label="am1bcc_charges1",
        )
        loss = target.evaluate_loss(dgl_batch, labels, predictions, {})
        assert torch.isclose(loss, torch.tensor([6.75]))


class TestHeavyAtomReadoutTarget:
    def test_single_molecule(self, dgl_methane):
        predictions = {
            "am1bcc_charges": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        }
        labels = {
            "am1bcc_charges": torch.tensor([[3.0]]),
        }

        target = HeavyAtomReadoutTarget(
            metric="rmse",
            target_label="am1bcc_charges",
            prediction_label="am1bcc_charges",
        )
        assert target.get_required_columns() == ["am1bcc_charges"]

        loss = target.evaluate_loss(dgl_methane, labels, predictions, {})
        assert torch.isclose(loss, torch.tensor([2.0]))


class TestSingleDipoleTarget:
    def test_single_molecule(self, dgl_methane):
        predictions = {
            "am1bcc_charges": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        }

        fake_conformers = torch.tensor(np.arange(15)).float()
        labels = {
            "am1bcc_charges": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),
            "am1bcc_dipoles": torch.tensor([[0.0, 10.0, 20.0]]),
            "conformers": fake_conformers,
        }

        target = SingleDipoleTarget(
            metric="mae",
            charge_label="am1bcc_charges",
            target_label="am1bcc_dipoles",
            conformation_column="conformers",
        )

        # expected_dipoles = np.array([120., 135., 150.])
        expected_columns = ["am1bcc_dipoles", "conformers"]
        assert target.get_required_columns() == expected_columns

        loss = target.evaluate_loss(dgl_methane, labels, predictions, {})
        assert torch.isclose(loss, torch.tensor([125.0]))


class TestMultipleDipoleTarget:
    def test_single_molecule(self, dgl_methane):
        predictions = {
            "am1bcc_charges": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        }

        fake_conformers = torch.tensor(np.arange(30)).float()
        n_conformers = torch.tensor(
            [
                2,
            ]
        )
        labels = {
            "am1bcc_charges": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),
            "am1bcc_dipoles": torch.tensor([[0.0, 10.0, 20.0, 30.0, 40.0, 50.0]]),
            "conformers": fake_conformers,
            "n_conformers": n_conformers,
        }

        target = MultipleDipoleTarget(
            metric="mae",
            charge_label="am1bcc_charges",
            target_label="am1bcc_dipoles",
            conformation_column="conformers",
            n_conformation_column="n_conformers",
        )

        # expected_dipoles = np.array([120., 135., 150., 345., 360., 375.])
        expected_columns = ["am1bcc_dipoles", "conformers", "n_conformers"]
        assert target.get_required_columns() == expected_columns

        loss = target.evaluate_loss(dgl_methane, labels, predictions, {})
        assert torch.isclose(loss, torch.tensor([222.5]))


class TestMultipleESPTarget:
    def test_single_molecule(self, dgl_methane):
        predictions = {
            "am1bcc_charges": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        }

        inv_dist = torch.tensor(
            [
                0.05773503,
                0.08247861,
                0.14433757,
                0.57735027,
                0.28867513,
                0.04441156,
                0.05773503,
                0.08247861,
                0.14433757,
                0.57735027,
                0.03608439,
                0.04441156,
                0.05773503,
                0.08247861,
                0.14433757,
                0.14433757,
                0.57735027,
                0.28867513,
                0.11547005,
                0.07216878,
                0.08247861,
                0.14433757,
                0.57735027,
                0.28867513,
                0.11547005,
            ]
        )
        reference_esps = np.array(
            [5.55905831, 4.77773209, 1.71476202, 4.18578945, 5.04356699]
        )
        n_conformers = torch.tensor(
            [
                2,
            ]
        )
        labels = {
            "am1bcc_charges": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),
            "am1bcc_esps": torch.tensor(reference_esps),
            "esp_grid_inverse_distances": inv_dist,
            "esp_lengths": torch.tensor([3, 2]),
            "n_conformers": n_conformers,
        }

        target = MultipleESPTarget(
            metric="mae",
            charge_label="am1bcc_charges",
            target_label="am1bcc_esps",
            inverse_distance_matrix_column="esp_grid_inverse_distances",
            esp_length_column="esp_lengths",
            n_esp_column="n_conformers",
        )

        # expected_dipoles = np.array([120., 135., 150., 345., 360., 375.])
        expected_columns = [
            "am1bcc_esps",
            "esp_grid_inverse_distances",
            "esp_lengths",
            "n_conformers",
        ]
        assert target.get_required_columns() == expected_columns

        loss = target.evaluate_loss(dgl_methane, labels, predictions, {})
        assert torch.isclose(loss, torch.tensor([0.965650046]))
