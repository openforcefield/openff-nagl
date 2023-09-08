import importlib_resources
import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from openff.units import unit

from openff.nagl.nn.gcn._sage import SAGEConvStack
from openff.nagl.nn._containers import ConvolutionModule, ReadoutModule
from openff.nagl.nn._models import BaseGNNModel, GNNModel
from openff.nagl.nn._pooling import PoolAtomFeatures, PoolBondFeatures
from openff.nagl.nn.postprocess import ComputePartialCharges
from openff.nagl.nn._sequential import SequentialLayers
from openff.nagl.tests.data.files import (
    EXAMPLE_AM1BCC_MODEL_STATE_DICT,
    MODEL_CONFIG_V7,
    EXAMPLE_AM1BCC_MODEL,
)
from openff.nagl.features.atoms import (
    AtomicElement,
    AtomConnectivity,
    AtomAverageFormalCharge,
    AtomInRingOfSize,
)


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


class TestGNNModel:
    @pytest.fixture()
    def am1bcc_model(self):
        model = GNNModel.from_yaml_file(MODEL_CONFIG_V7)
        model.load_state_dict(torch.load(EXAMPLE_AM1BCC_MODEL_STATE_DICT))
        model.eval()

        return model

    def test_init(self):
        from openff.nagl.features import atoms, bonds

        atom_features = (
            atoms.AtomicElement(["C", "O", "H", "N", "S", "F", "Br", "Cl", "I", "P"]),
            atoms.AtomConnectivity(),
            atoms.AtomAverageFormalCharge(),
            atoms.AtomHybridization(),
            atoms.AtomInRingOfSize(3),
            atoms.AtomInRingOfSize(4),
            atoms.AtomInRingOfSize(5),
            atoms.AtomInRingOfSize(6),
        )

        bond_features = (
            bonds.BondInRingOfSize(3),
            bonds.BondInRingOfSize(4),
            bonds.BondInRingOfSize(5),
            bonds.BondInRingOfSize(6),
        )

        model = GNNModel(
            convolution_architecture="SAGEConv",
            n_convolution_hidden_features=128,
            n_convolution_layers=3,
            n_readout_hidden_features=128,
            n_readout_layers=4,
            activation_function="ReLU",
            postprocess_layer="compute_partial_charges",
            readout_name=f"am1bcc-charges",
            learning_rate=0.001,
            atom_features=atom_features,
            bond_features=bond_features,
        )

    def test_to_nagl(self, am1bcc_model):
        pytest.importorskip("dgl")
        assert am1bcc_model._is_dgl
        nagl_model = am1bcc_model._as_nagl()

        original_state_dict = am1bcc_model.state_dict()
        new_state_dict = nagl_model.state_dict()
        assert original_state_dict.keys() == new_state_dict.keys()
        for key in original_state_dict.keys():
            assert torch.allclose(original_state_dict[key], new_state_dict[key])

    def test_compute_property_dgl(self, am1bcc_model, openff_methane_uncharged):
        pytest.importorskip("dgl")
        charges = am1bcc_model._compute_property_dgl(openff_methane_uncharged)
        charges = charges.detach().numpy().flatten()
        expected = np.array([-0.143774, 0.035943, 0.035943, 0.035943, 0.035943])
        assert_allclose(charges, expected, atol=1e-5)

    def test_compute_property_networkx(self, am1bcc_model, openff_methane_uncharged):
        charges = am1bcc_model._compute_property_nagl(openff_methane_uncharged)
        charges = charges.detach().numpy().flatten()
        expected = np.array([-0.143774, 0.035943, 0.035943, 0.035943, 0.035943])
        assert_allclose(charges, expected, atol=1e-5)

    def test_compute_property(self, am1bcc_model, openff_methane_uncharged):
        charges = am1bcc_model.compute_property(openff_methane_uncharged, as_numpy=True)
        expected = np.array([-0.143774, 0.035943, 0.035943, 0.035943, 0.035943])
        assert_allclose(charges, expected, atol=1e-5)

    def test_load(self, openff_methane_uncharged):
        model = GNNModel.load(EXAMPLE_AM1BCC_MODEL, eval_mode=True)
        assert isinstance(model, GNNModel)

        assert model.atom_features == [
            AtomicElement(
                categories=["C", "O", "H", "N", "S", "F", "Br", "Cl", "I", "P"]
            ),
            AtomConnectivity(categories=[1, 2, 3, 4]),
            AtomAverageFormalCharge(),
            AtomInRingOfSize(ring_size=3),
            AtomInRingOfSize(ring_size=4),
            AtomInRingOfSize(ring_size=5),
            AtomInRingOfSize(ring_size=6),
        ]

        charges = model.compute_property(openff_methane_uncharged, as_numpy=True)
        expected = np.array([-0.111393, 0.027848, 0.027848, 0.027848, 0.027848])
        assert_allclose(charges, expected, atol=1e-5)

    @pytest.mark.parametrize(
        "smiles",
        [
            "CNC",
            "CCNCO",
            "CCO",
            "CC(=O)O",
            "CC(=O)([O-])",
            "C1CC1",
            "C1CCC1",
            "C1CNCC1",
            "C1CNC(=O)CC1",
            "FN(F)C(F)(F)N(F)C(N(F)F)(N(F)F)N(F)F",
            "CC(C)(C)c1s[n-]c(=O)c1C[C@H]([NH3+])C(=O)O",
            "O=C[O-]",
            "CCCCCCOP(=O)([O-])OCC[NH2+]C",
            "C[NH2+]CC(=O)N(C)CC(=O)[O-]",
            "O=C(CC(O)(C(F)(F)F)C(F)(F)F)C1=CC(=S(=O)=O)CC(F)=C1",
            "O=c1[nH]cnc2c([NH2+]CCCP(=O)([O-])[O-])c[nH]c12",
            "CNC(=O)[C@H](C[S-])NC(=O)[C@@H](Cc1c[nH]c[nH+]1)NC(C)=O",
            "CNC(=O)[C@H](C[S-])NC(=O)[C@@H](CCCC[NH3+])NC(C)=O",
            "[NH3+][C@@H]1C(=C(F)F)CC[C@@H]1C(=O)[O-]",
            "[NH3+][C@H](CCC(=O)[O-])C(=O)OC[C@H]1CCC[C@@H](CO)N1",
            "CNC(=O)[C@@H](CCCC[NH3+])NC(=O)[C@H](C[S-])NC(C)=O",
            "CC(C)C[C@H](NC(=O)[P@](=O)(O)[C@H]([NH3+])CC(C)C)C(=O)[O-]",
            "Cc1nc(-c2ccc(NS(=O)(=O)C3=CSC=C=C3Cl)cc2)[nH]c2nnc(N)c1-2",
            "C=C(C[N+](=O)[O-])N[C@@H](CCCCNc1ccc([N+](=O)[O-])cc1[NH+](O)O)C(=O)O",
            "Fc1ncccc1[C@H]1CCC2=C(C1)C(=C1N=NN=N1)N=N2",
            "CN1c2ccccc2[C@@H](NCCCCC(=O)O)c2ccc(Cl)cc2S1([O-])[O-]",
            "CC1=C(N2CCN(C3=C(C)N([O-])ON3)CC2)NON1[O-]",
            "[O-]S(O)(O)CC[NH+]1CCOCC1",
            "O=NN([O-])[O-]",
            "[O-]P([O-])[O-]",
            "C#N"
        ],
    )
    def test_load_and_compute(self, smiles):
        from openff.toolkit import Molecule

        model = GNNModel.load(EXAMPLE_AM1BCC_MODEL, eval_mode=True)
        testdir = importlib_resources.files("openff.nagl") / "tests"
        path = testdir / "data" / "example_am1bcc_sage_charges" / f"{smiles}.sdf"
        molecule = Molecule.from_file(
            str(path), file_format="sdf", allow_undefined_stereo=True
        )

        desired = molecule.partial_charges.m_as(unit.elementary_charge)
        computed = model.compute_property(molecule, as_numpy=True)
        assert_allclose(computed, desired, atol=1e-5)

    def test_save(self, am1bcc_model, openff_methane_uncharged, tmpdir):
        with tmpdir.as_cwd():
            am1bcc_model.save("model.pt")
            model = GNNModel.load("model.pt", eval_mode=True)
            charges = model.compute_property(openff_methane_uncharged, as_numpy=True)
            expected = np.array([-0.143774, 0.035943, 0.035943, 0.035943, 0.035943])
            assert_allclose(charges, expected, atol=1e-5)
