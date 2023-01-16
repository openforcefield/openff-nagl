import numpy as np
import pytest
import torch

from openff.nagl.nn.gcn._sage import SAGEConvStack
from openff.nagl.nn._containers import ConvolutionModule, ReadoutModule
from openff.nagl.nn._models import BaseGNNModel
from openff.nagl.nn._pooling import PoolAtomFeatures, PoolBondFeatures
from openff.nagl.nn.postprocess import ComputePartialCharges
from openff.nagl.nn._sequential import SequentialLayers


@pytest.fixture()
def mock_atom_model() -> BaseGNNModel:
    convolution = ConvolutionModule(
        n_input_features=4,
        hidden_feature_sizes=[4],
        architecture="SAGEConv",
    )
    readout_layers = SequentialLayers.with_layers(
        n_input_features=4,
        hidden_feature_sizes=[2],
    )
    model = BaseGNNModel(
        convolution_module=convolution,
        readout_modules={
            "atom": ReadoutModule(
                pooling_layer=PoolAtomFeatures(),
                readout_layers=readout_layers,
                postprocess_layer=ComputePartialCharges(),
            ),
        },
        learning_rate=0.01,
    )
    return model


class TestBaseGNNModel:
    def test_init(self):
        model = BaseGNNModel(
            convolution_module=ConvolutionModule(
                n_input_features=1,
                hidden_feature_sizes=[2, 3],
                architecture="SAGEConv",
            ),
            readout_modules={
                "atom": ReadoutModule(
                    pooling_layer=PoolAtomFeatures(),
                    readout_layers=SequentialLayers.with_layers(
                        n_input_features=2,
                        hidden_feature_sizes=[2],
                        layer_activation_functions=["Identity"],
                    ),
                    postprocess_layer=ComputePartialCharges(),
                ),
                "bond": ReadoutModule(
                    pooling_layer=PoolBondFeatures(
                        layers=SequentialLayers.with_layers(
                            n_input_features=4,
                            hidden_feature_sizes=[4],
                        )
                    ),
                    readout_layers=SequentialLayers.with_layers(
                        n_input_features=4,
                        hidden_feature_sizes=[8],
                    ),
                ),
            },
            learning_rate=0.01,
        )

        assert model.convolution_module is not None
        assert isinstance(model.convolution_module, ConvolutionModule)

        assert isinstance(model.convolution_module.gcn_layers, SAGEConvStack)
        assert len(model.convolution_module.gcn_layers) == 2

        readouts = model.readout_modules
        assert all(x in readouts for x in ["atom", "bond"])

        assert isinstance(readouts["atom"].pooling_layer, PoolAtomFeatures)
        assert isinstance(readouts["bond"].pooling_layer, PoolBondFeatures)

        assert np.isclose(model.learning_rate, 0.01)

    def test_forward(self, mock_atom_model, dgl_methane):
        output = mock_atom_model.forward(dgl_methane)
        assert "atom" in output
        assert output["atom"].shape == (5, 1)

    @pytest.mark.parametrize(
        "method_name", ["training_step", "validation_step", "test_step"]
    )
    def test_step(self, mock_atom_model, method_name, dgl_methane, monkeypatch):
        def mock_forward(_):
            return {"atom": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])}

        monkeypatch.setattr(mock_atom_model, "forward", mock_forward)

        loss_function = getattr(mock_atom_model, method_name)
        fake_comparison = {"atom": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]])}
        loss = list(loss_function((dgl_methane, fake_comparison), 0).values())[0]
        assert torch.isclose(loss, torch.tensor([1.0]))

    def test_configure_optimizers(self, mock_atom_model):
        optimizer = mock_atom_model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        assert torch.isclose(torch.tensor(optimizer.defaults["lr"]), torch.tensor(0.01))

