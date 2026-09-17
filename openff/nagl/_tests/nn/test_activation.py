import torch
import torch.nn.functional as F

import pytest

from openff.nagl.nn.activation import ActivationFunction

@pytest.mark.parametrize("name", ["relu", "ReLU"])
def test_get(name):
    assert ActivationFunction.get(name) == ActivationFunction.ReLU


def test_get_invalid():
    with pytest.raises(KeyError):
        ActivationFunction.get("invalid")


@pytest.mark.parametrize("name", ["relu", "ReLU"])
def test_get_value(name):
    value = ActivationFunction.get_value(name)
    assert isinstance(value, torch.nn.ReLU)


def test_get_value_invalid():
    with pytest.raises(KeyError):
        ActivationFunction.get_value("invalid")


def test_get__function():
    assert ActivationFunction.get_function("relu") == F.relu