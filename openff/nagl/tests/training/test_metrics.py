import torch

from openff.nagl.training.metrics import (
    RMSEMetric,
    MSEMetric,
    MAEMetric,
)

def test_rmse():
    metric = RMSEMetric()
    predicted = torch.tensor([-1., 3.])
    expected = torch.tensor([0., 10.])

    calculated = metric(predicted, expected)
    reference = torch.tensor(5.0)  # sqrt( (1 + 49)/2 )
    assert torch.isclose(calculated, reference)

def test_mse():
    metric = MSEMetric()
    predicted = torch.tensor([-1., 3.])
    expected = torch.tensor([0., 10.])

    calculated = metric(predicted, expected)
    reference = torch.tensor(25.0)
    assert torch.isclose(calculated, reference)

def test_mae():
    metric = MAEMetric()
    predicted = torch.tensor([-1., 3.])
    expected = torch.tensor([0., 10.])

    calculated = metric(predicted, expected)
    reference = torch.tensor(4.0)
    assert torch.isclose(calculated, reference)
