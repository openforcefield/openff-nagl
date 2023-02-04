import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest
import torch

from openff.utilities import requires_package
from openff.nagl.nn.activation import ActivationFunction
from openff.nagl.nn.gcn import SAGEConvStack
from openff.nagl.nn.gcn._sage import SAGEConv

try:
    import dgl

    _BASE_SAGECONV_CLASS = dgl.nn.pytorch.SAGEConv
except ImportError:
    _BASE_SAGECONV_CLASS = SAGEConv


class TestDGLSAGEConvStack:
    def test_default_with_layers(self):
        stack = SAGEConvStack.with_layers(
            n_input_features=1,
            hidden_feature_sizes=[2, 3],
        )
        stack.reset_parameters()

        assert len(stack) == 2
        assert all(isinstance(layer, _BASE_SAGECONV_CLASS) for layer in stack)

        first, second = stack
        assert np.isclose(first.feat_drop.p, 0.0)
        assert first.fc_self.in_features == 1
        assert first.fc_self.out_features == 2

        assert np.isclose(second.feat_drop.p, 0.0)
        assert second.fc_self.in_features == 2
        assert second.fc_self.out_features == 3

    def test_with_layers_inputs(self):
        stack = SAGEConvStack.with_layers(
            n_input_features=2,
            hidden_feature_sizes=[3],
            layer_activation_functions=[ActivationFunction.LeakyReLU],
            layer_dropout=[0.5],
            layer_aggregator_types=["lstm"],
        )

        assert len(stack) == 1
        assert all(isinstance(layer, _BASE_SAGECONV_CLASS) for layer in stack)

        layer = stack[0]
        assert np.isclose(layer.feat_drop.p, 0.5)
        assert layer.lstm.input_size == 2
        assert layer.lstm.hidden_size == 2
        assert layer.fc_neigh.out_features == 3
        assert isinstance(layer.activation, torch.nn.LeakyReLU)

    def test_forward(self, dgl_methane):
        stack = SAGEConvStack.with_layers(
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
            SAGEConvStack.with_layers(
                n_input_features=1,
                hidden_feature_sizes=[2, 3],
                layer_dropout=[0.5],
            )


class TestDGLSageConv:
    def test_forward_values(self, dgl_methane):
        dgl = pytest.importorskip("dgl")
        layer = dgl.nn.pytorch.SAGEConv(
            in_feats=4,
            out_feats=3,
            aggregator_type="mean",
            feat_drop=0,
            activation=torch.nn.Sigmoid(),
            bias=False,
        )

        layer.fc_neigh.weight.data.fill_(1.0)
        layer.fc_self.weight.data.fill_(2.0)

        expected_features = np.array(
            [
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        assert_allclose(dgl_methane.atom_features.detach().numpy(), expected_features)

        results = layer.forward(dgl_methane.homograph, dgl_methane.atom_features)
        results = results.detach().numpy()
        assert results.shape == (5, 3)
        assert_array_almost_equal(results, 0.952574)


class TestSageConv:
    @pytest.fixture()
    def sageconv_layer(self):
        layer = SAGEConv(
            in_feats=4,
            out_feats=3,
            aggregator_type="mean",
            feat_drop=0,
            activation=torch.nn.Sigmoid(),
            bias=False,
        )

        layer.fc_neigh.weight.data.fill_(1.0)
        layer.fc_self.weight.data.fill_(2.0)
        return layer

    def test_forward_values_dgl(self, sageconv_layer, dgl_methane):
        expected_features = np.array(
            [
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        assert_allclose(dgl_methane.atom_features.detach().numpy(), expected_features)

        results = sageconv_layer.forward(
            dgl_methane.homograph, dgl_methane.atom_features
        )
        results = results.detach().numpy()
        assert results.shape == (5, 3)
        assert_array_almost_equal(results, 0.952574)

    def test_forward_values_dgl(self, sageconv_layer, nx_methane):
        expected_features = np.array(
            [
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        assert_allclose(nx_methane.atom_features.detach().numpy(), expected_features)

        results = sageconv_layer.forward(nx_methane.homograph, nx_methane.atom_features)
        results = results.detach().numpy()
        assert results.shape == (5, 3)
        assert_array_almost_equal(results, 0.952574)
