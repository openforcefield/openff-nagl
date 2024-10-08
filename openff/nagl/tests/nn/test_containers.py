import torch

from openff.nagl.nn.activation import ActivationFunction
from openff.nagl.nn.gcn import SAGEConvStack
from openff.nagl.nn._containers import (
    ConvolutionModule,
    ReadoutModule,
)
from openff.nagl.nn._pooling import PoolAtomFeatures
from openff.nagl.nn.postprocess import ComputePartialCharges
from openff.nagl.nn._sequential import SequentialLayers


class TestConvolutionModule:
    def test_init(self):
        module = ConvolutionModule(
            n_input_features=2,
            hidden_feature_sizes=[2, 2],
            architecture="SAGEConv",
            layer_activation_functions="ReLU",
        )
        assert isinstance(module.gcn_layers, SAGEConvStack)
        assert len(module.gcn_layers) == 2

        for layer in module.gcn_layers:
            assert isinstance(layer.activation, torch.nn.ReLU)


class TestReadoutModule:
    def test_init(self):
        sequential = SequentialLayers.with_layers(
            n_input_features=1,
            hidden_feature_sizes=[1],
        )
        module = ReadoutModule(PoolAtomFeatures(), sequential, ComputePartialCharges())
        assert isinstance(module.pooling_layer, PoolAtomFeatures)
        assert isinstance(module.readout_layers, SequentialLayers)
        assert isinstance(module.postprocess_layer, ComputePartialCharges)
