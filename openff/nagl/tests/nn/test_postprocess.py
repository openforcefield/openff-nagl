import numpy as np
import pytest
import torch

from openff.nagl.molecule._dgl import DGLMolecule, DGLMoleculeBatch
from openff.nagl.nn.postprocess import (
    ComputePartialCharges,
    RegularizedComputePartialCharges,
)

# @pytest.fixture
# def dgl_carboxylate


def test_calculate_partial_charges_neutral():
    charges = ComputePartialCharges._calculate_partial_charges(
        electronegativity=torch.tensor([30.8, 27.4, 27.4, 27.4, 27.4]),
        hardness=torch.tensor([78.4, 73.9, 73.9, 73.9, 73.9]),
        total_charge=0.0,
    ).numpy()

    assert np.isclose(charges.sum(), 0.0)
    expected = np.array(
        [-0.03509676, 0.00877419, 0.00877419, 0.00877419, 0.00877419]
    ).reshape((-1, 1))
    assert np.allclose(charges, expected)


@pytest.mark.parametrize(
    "q0, qi",
    [
        (
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.03509676, 0.00877419, 0.00877419, 0.00877419, 0.00877419],
        ),
        (
            [-0.04, 0.01, 0.01, 0.01, 0.01],
            [-0.07509676, 0.01877419, 0.01877419, 0.01877419, 0.01877419],
        ),
    ],
)
def test_regularized_calculate_partial_charges_neutral(q0, qi):
    charges = RegularizedComputePartialCharges._calculate_partial_charges(
        charge_priors=torch.tensor(q0),
        electronegativity=torch.tensor([30.8, 27.4, 27.4, 27.4, 27.4]),
        hardness=torch.tensor([78.4, 73.9, 73.9, 73.9, 73.9]),
        total_charge=0.0,
    ).numpy()
    expected = np.array(qi).reshape((-1, 1))
    assert np.allclose(charges, expected)


def test_calculate_partial_charges_charged():
    charges = ComputePartialCharges._calculate_partial_charges(
        electronegativity=torch.tensor([30.8, 49.3, 27.4, 27.4, 27.4]),
        hardness=torch.tensor([78.4, 25.0, 73.9, 73.9, 73.9]),
        total_charge=-1.0,
    ).numpy()

    assert np.isclose(charges.sum(), -1.0)
    expected = np.array(
        [-0.05438471, -0.91055036, -0.01168823, -0.01168823, -0.01168823]
    ).reshape((-1, 1))
    assert np.allclose(charges, expected)


@pytest.mark.parametrize(
    "q0, qi",
    [
        (
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.22580644, -0.1935484, -0.1935484, -0.1935484, -0.1935484],
        ),
        (
            [-0.04, 0.01, 0.01, 0.01, 0.01],
            [-0.26580644, -0.1835484, -0.1835484, -0.1835484, -0.1835484],
        ),
    ],
)
def test_regularized_calculate_partial_charges_charged(q0, qi):
    charges = RegularizedComputePartialCharges._calculate_partial_charges(
        charge_priors=torch.tensor(q0),
        electronegativity=torch.tensor([30.8, 27.4, 27.4, 27.4, 27.4]),
        hardness=torch.tensor([78.4, 73.9, 73.9, 73.9, 73.9]),
        total_charge=-1.0,
    ).numpy()
    expected = np.array(qi).reshape((-1, 1))
    assert np.allclose(charges, expected)


def test_compute_charges_forward(dgl_methane):
    inputs = torch.tensor(
        [
            [30.8, 78.4],
            [27.4, 73.9],
            [27.4, 73.9],
            [27.4, 73.9],
            [27.4, 73.9],
        ]
    )
    charges = ComputePartialCharges().forward(dgl_methane, inputs)
    assert np.isclose(charges.sum(), 0.0)
    expected = np.array(
        [
            -0.0351,
            0.0088,
            0.0088,
            0.0088,
            0.0088,
        ]
    ).reshape((-1, 1))
    assert np.allclose(charges, expected, atol=1e-4)


def test_regularized_compute_charges_forward(dgl_methane):
    inputs = torch.tensor(
        [
            [-0.1, 30.8, 78.4],
            [-0.2, 27.4, 73.9],
            [0.3, 27.4, 73.9],
            [-0.5, 27.4, 73.9],
            [0.5, 27.4, 73.9],
        ]
    )
    charges = RegularizedComputePartialCharges().forward(dgl_methane, inputs)
    assert np.isclose(charges.sum(), 0.0)
    expected = np.array(
        [
            -0.1351,
            -0.1912,
            0.3088,
            -0.4912,
            0.5088,
        ]
    ).reshape((-1, 1))
    assert np.allclose(charges, expected, atol=1e-4)


def test_compute_charges_forward_batched(openff_carboxylate):
    pytest.importorskip("dgl")
    dgl_carboxylate = DGLMolecule.from_openff(
        openff_carboxylate,
        enumerate_resonance_forms=True,
        lowest_energy_only=True,
    )
    dgl_hcl = DGLMolecule.from_smiles("[H]Cl")
    batch = DGLMoleculeBatch.from_dgl_molecules([dgl_carboxylate, dgl_hcl])
    inputs = torch.tensor(
        [
            # [H]C(=O)O- form 1
            [30.0, 80.0],
            [35.0, 75.0],
            [40.0, 70.0],
            [50.0, 65.0],
            # [H]C(=O)O- form 2
            [30.0, 80.0],
            [35.0, 75.0],
            [50.0, 65.0],
            [40.0, 70.0],
            # [H]Cl
            [55.0, 60.0],
            [60.0, 55.0],
        ]
    )
    partial_charges = ComputePartialCharges().forward(batch, inputs)
    assert partial_charges.shape == (6, 1)

    assert np.isclose(partial_charges.sum(), -1.0)
    # The carboxylate oxygen charges should be identical.
    assert np.allclose(partial_charges[2], partial_charges[3])

    expected = np.array(
        [
            [-0.1087],
            [-0.1826],
            [-0.3543],
            [-0.3543],
            [0.0435],
            [-0.0435],
        ]
    )

    assert np.allclose(partial_charges, expected, atol=1e-4)


def test_regularized_compute_charges_forward_batched(openff_carboxylate):
    pytest.importorskip("dgl")
    dgl_carboxylate = DGLMolecule.from_openff(
        openff_carboxylate,
        enumerate_resonance_forms=True,
        lowest_energy_only=True,
    )
    dgl_hcl = DGLMolecule.from_smiles("[H]Cl")
    batch = DGLMoleculeBatch.from_dgl_molecules([dgl_carboxylate, dgl_hcl])
    inputs = torch.tensor(
        [
            # [H]C(=O)O- form 1
            [0.1, 30.0, 80.0],
            [-0.2, 35.0, 75.0],
            [-0.3, 40.0, 70.0],
            [0.3, 50.0, 65.0],
            # [H]C(=O)O- form 2
            [0.1, 30.0, 80.0],
            [-0.2, 35.0, 75.0],
            [0.3, 50.0, 65.0],
            [-0.3, 40.0, 70.0],
            # [H]Cl
            [0.1, 55.0, 60.0],
            [-0.1, 60.0, 55.0],
        ]
    )
    partial_charges = ComputePartialCharges().forward(batch, inputs)
    assert partial_charges.shape == (6, 1)

    assert np.isclose(partial_charges.sum(), -1.0)
    # The carboxylate oxygen charges should be identical.
    assert np.allclose(partial_charges[2], partial_charges[3])

    expected = np.array(
        [
            [-0.3163],
            [-0.2626],
            [-0.2105],
            [-0.2105],
            [-0.0017],
            [0.0017],
        ]
    )

    assert np.allclose(partial_charges, expected, atol=1e-4)
