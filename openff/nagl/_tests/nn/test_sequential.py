import numpy as np
import pytest
import torch

from openff.nagl.nn._sequential import SequentialLayers


class TestSequentialLayers:
    def test_init_default(self):
        sequential_layers = SequentialLayers.with_layers(
            n_input_features=1,
            hidden_feature_sizes=[2],
        )
        assert len(sequential_layers) == 3
        assert isinstance(sequential_layers[0], torch.nn.Linear)
        assert isinstance(sequential_layers[1], torch.nn.ReLU)
        assert isinstance(sequential_layers[2], torch.nn.Dropout)
        assert np.isclose(sequential_layers[2].p, 0.0)

    def test_init_with_inputs(self):
        sequential_layers = SequentialLayers.with_layers(
            n_input_features=1,
            hidden_feature_sizes=[2, 1],
            layer_activation_functions="ReLU",
            layer_dropout=[0.0, 0.5],
        )
        assert len(sequential_layers) == 6
        assert isinstance(sequential_layers[0], torch.nn.Linear)
        assert isinstance(sequential_layers[1], torch.nn.ReLU)
        assert isinstance(sequential_layers[2], torch.nn.Dropout)
        assert np.isclose(sequential_layers[2].p, 0.0)

        assert isinstance(sequential_layers[3], torch.nn.Linear)
        assert isinstance(sequential_layers[4], torch.nn.ReLU)
        assert isinstance(sequential_layers[5], torch.nn.Dropout)
        assert np.isclose(sequential_layers[5].p, 0.5)

    def test_invalid_lengths(self):
        expected_err = (
            r"`layer_activation_functions` \(length 2\) must be a list of same length "
            r"as `hidden_feature_sizes` \(length 1\)."
        )
        with pytest.raises(ValueError, match=expected_err):
            SequentialLayers.with_layers(
                n_input_features=1,
                hidden_feature_sizes=[2],
                layer_activation_functions=["ReLU", "LeakyRELU"],
            )
