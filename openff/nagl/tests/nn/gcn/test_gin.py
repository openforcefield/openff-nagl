import numpy as np
import pytest
import torch

from openff.nagl.nn.activation import ActivationFunction
from openff.nagl.nn.gcn._gin import GINConvStack, GINConv, DGLGINConv

try:
    import dgl

    _BASE_GINCONV_CLASS = DGLGINConv
except ImportError:
    _BASE_GINCONV_CLASS = GINConv


class TestGINConvStack:
    def test_default_with_layers(self):
        stack = GINConvStack.with_layers(
            n_input_features=1,
            hidden_feature_sizes=[2, 3],
        )
        stack.reset_parameters()

        assert len(stack) == 2
        assert all(isinstance(layer, _BASE_GINCONV_CLASS) for layer in stack)

        first, second = stack
        assert np.isclose(first.feat_drop.p, 0.0)
        assert first.fc_self.in_features == 1
        assert first.fc_self.out_features == 2

        assert np.isclose(second.feat_drop.p, 0.0)
        assert second.fc_self.in_features == 2
        assert second.fc_self.out_features == 3

    def test_with_layers_inputs(self):
        stack = GINConvStack.with_layers(
            n_input_features=2,
            hidden_feature_sizes=[3],
            layer_activation_functions=[ActivationFunction.LeakyReLU],
            layer_dropout=[0.5],
            layer_aggregator_types=["sum"],
        )

        assert len(stack) == 1
        assert all(isinstance(layer, _BASE_GINCONV_CLASS) for layer in stack)

        layer = stack[0]
        assert np.isclose(layer.feat_drop.p, 0.5)
        assert isinstance(layer.activation, torch.nn.LeakyReLU)

    def test_forward(self, dgl_methane):
        stack = GINConvStack.with_layers(
            n_input_features=4,
            hidden_feature_sizes=[2],
        )

        h = stack.forward(dgl_methane.homograph, dgl_methane.atom_features)
        assert h.detach().numpy().shape == (5, 2)

    def test_invalid_lengths(self):
        expected_err = (
            r"`layer_dropout` \(length 1\) must be a list of same length "
            r"as `hidden_feature_sizes` \(length 2\)."
        )
        with pytest.raises(ValueError, match=expected_err):
            GINConvStack.with_layers(
                n_input_features=1,
                hidden_feature_sizes=[2, 3],
                layer_dropout=[0.5],
            )
