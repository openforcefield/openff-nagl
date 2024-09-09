import typing
from typing import Dict, List
import pytest
from openff.nagl.nn._containers import ReadoutModule
import torch
import numpy as np

from openff.nagl.training.metrics import RMSEMetric
from openff.nagl.training.loss import (
    _BaseTarget,
    MultipleDipoleTarget,
    SingleDipoleTarget,
    HeavyAtomReadoutTarget,
    ReadoutTarget,
    MultipleESPTarget,
    GeneralLinearFitTarget
)

class TestBaseTarget:

    class BaseTarget(_BaseTarget):
        name: typing.Literal["base"]
        def get_required_columns(self) -> List[str]:
            return []

        def evaluate_target(self, molecules, labels, predictions, readout_modules) -> "torch.Tensor":
            return torch.tensor([0.0])

    def test_validate_metric(self):
        input_text = '{"metric": "rmse", "name": "readout", "prediction_label": "charges", "target_label": "charges"}'
        target = ReadoutTarget.parse_raw(input_text)
        assert isinstance(target.metric, RMSEMetric)

    def test_non_implemented_methods(self):
        target = self.BaseTarget(name="base", metric="rmse", target_label="charges")
        with pytest.raises(NotImplementedError):
            target.compute_reference(None)
        with pytest.raises(NotImplementedError):
            target.report_artifact(None, None, None, None)

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

        loss = target.evaluate_loss(
            dgl_methane,
            labels,
            predictions,
            {}
        )
        assert torch.isclose(loss, torch.tensor([1.0]))

    def test_batch(self, dgl_batch):
        predictions = {
            "am1bcc_charges1": torch.tensor(np.arange(28)).float()
        }
        labels = {
            "am1bcc_charges2": torch.tensor(np.arange(28) / 2).float()
        }

        target = ReadoutTarget(
            metric="mae",
            target_label="am1bcc_charges2",
            prediction_label="am1bcc_charges1",
        )
        loss = target.evaluate_loss(
            dgl_batch,
            labels,
            predictions,
            {}
        )
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

        loss = target.evaluate_loss(
            dgl_methane,
            labels,
            predictions,
            {}
        )
        assert torch.isclose(loss, torch.tensor([2.0]))


class TestSingleDipoleTarget:

    def test_single_molecule(self, dgl_methane):
        predictions = {
            "am1bcc_charges": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        }

        fake_conformers = torch.tensor(np.arange(15)).float()
        labels = {
            "am1bcc_charges": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),
            "am1bcc_dipoles": torch.tensor([[0.0, 10., 20.]]),
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

        loss = target.evaluate_loss(
            dgl_methane,
            labels,
            predictions,
            {}
        )
        assert torch.isclose(loss, torch.tensor([125.0]))


class TestMultipleDipoleTarget:

    def test_single_molecule(self, dgl_methane):
        predictions = {
            "am1bcc_charges": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        }

        fake_conformers = torch.tensor(np.arange(30)).float()
        n_conformers = torch.tensor([2,])
        labels = {
            "am1bcc_charges": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),
            "am1bcc_dipoles": torch.tensor([[0.0, 10., 20., 30., 40., 50.]]),
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

        loss = target.evaluate_loss(
            dgl_methane,
            labels,
            predictions,
            {}
        )
        assert torch.isclose(loss, torch.tensor([222.5]))


class TestMultipleESPTarget:
    def test_single_molecule(self, dgl_methane):
        predictions = {
            "am1bcc_charges": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        }

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
        n_conformers = torch.tensor([2,])
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

        loss = target.evaluate_loss(
            dgl_methane,
            labels,
            predictions,
            {}
        )
        assert torch.isclose(loss, torch.tensor([0.965650046]))



class TestGeneralLinearFitTarget:
    def test_single_molecule(self, dgl_methane):
        predictions = {
            "am1bcc_charges": torch.tensor([-1, 3, -0.5, 2.1, 0.3]),
        }
        A_flat = torch.tensor(np.arange(25, dtype=float))
        labels={
            "design_matrix": A_flat,
            "calculated_vectors": torch.tensor([1, 2, 3, 4, 5])
        }
        target = GeneralLinearFitTarget(
            metric="rmse",
            prediction_label="am1bcc_charges",
            target_label="calculated_vectors",
            design_matrix_column="design_matrix",
        )

        required_columns = ["calculated_vectors", "design_matrix"]
        assert target.get_required_columns() == required_columns

        evaluated = target.evaluate_target(
            dgl_methane,
            labels,
            predictions,
            {}
        )
        assert torch.allclose(evaluated, torch.tensor([ 9.5, 29. , 48.5, 68. , 87.5]))

        loss = target.evaluate_loss(
            dgl_methane,
            labels=labels,
            predictions=predictions,
            readout_modules={},
        )
        assert torch.allclose(loss, torch.tensor([52.48571234]))
    

    def test_multiple_molecules(self, dgl_batch):
        charges = torch.cat([
            torch.arange(5),
            torch.arange(4),
            torch.arange(14),
        ]).float()
        predictions = {"charges": charges}

        A_flat = torch.cat([
            torch.arange(25),
            torch.arange(16),
            torch.arange(14 * 14),
        ]).float()
        calculated_vectors = torch.arange(23).float()
        labels = {
            "design_matrix": A_flat,
            "calculated_vectors": calculated_vectors,
        }

        target = GeneralLinearFitTarget(
            metric="rmse",
            prediction_label="charges",
            target_label="calculated_vectors",
            design_matrix_column="design_matrix",
        )

        evaluated = target.evaluate_target(
            dgl_batch,
            labels,
            predictions,
            {}
        )
        assert len(evaluated) == 23
        expected = torch.tensor([
            30,  80, 130, 180, 230,
            14, 38, 62, 86,
            819,  2093,  3367,  4641,  5915,  7189,  8463, 
            9737, 11011, 12285, 13559, 14833, 16107, 17381
        ]).float()
        assert torch.allclose(evaluated, expected)

        loss = target.evaluate_loss(
            dgl_batch,
            labels=labels,
            predictions=predictions,
            readout_modules={},
        )
        assert torch.allclose(loss, torch.tensor([8140.5599]))
