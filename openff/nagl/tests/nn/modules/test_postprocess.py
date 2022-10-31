import numpy as np
import torch

from openff.nagl.dgl import DGLMolecule, DGLMoleculeBatch
from openff.nagl.nn.modules.postprocess import ComputePartialCharges

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


def test_compute_charges_forward_batched(openff_carboxylate):
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
