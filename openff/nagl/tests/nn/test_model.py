import importlib_resources
import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from openff.toolkit import Molecule
from openff.units import unit

from openff.nagl.nn.gcn._sage import SAGEConvStack
from openff.nagl.nn._containers import ConvolutionModule, ReadoutModule
from openff.nagl.nn._models import BaseGNNModel, GNNModel
from openff.nagl.nn._pooling import PoolAtomFeatures, PoolBondFeatures
from openff.nagl.nn.postprocess import ComputePartialCharges
from openff.nagl.nn._sequential import SequentialLayers
from openff.nagl.domains import ChemicalDomain
from openff.nagl.tests.data.files import (
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
        )

        assert model.convolution_module is not None
        assert isinstance(model.convolution_module, ConvolutionModule)

        assert isinstance(model.convolution_module.gcn_layers, SAGEConvStack)
        assert len(model.convolution_module.gcn_layers) == 2

        readouts = model.readout_modules
        assert all(x in readouts for x in ["atom", "bond"])

        assert isinstance(readouts["atom"].pooling_layer, PoolAtomFeatures)
        assert isinstance(readouts["bond"].pooling_layer, PoolBondFeatures)


    def test_forward(self, mock_atom_model, dgl_methane):
        output = mock_atom_model.forward(dgl_methane)
        assert "atom" in output
        assert output["atom"].shape == (5, 1)

    

class TestGNNModel:
    @pytest.fixture()
    def am1bcc_model(self):
        model = GNNModel.load(EXAMPLE_AM1BCC_MODEL, eval_mode=True)
        model.chemical_domain = ChemicalDomain(
            allowed_elements=(1, 6)
        )
        # model = GNNModel.from_yaml(MODEL_CONFIG_V7)
        # model.load_state_dict(torch.load(EXAMPLE_AM1BCC_MODEL_STATE_DICT))
        # model.eval()

        return model

    @pytest.fixture()
    def expected_methane_charges(self):
        return np.array([-0.087334,  0.021833,  0.021833,  0.021833,  0.021833],)

    def test_init(self):
        from openff.nagl.features import atoms, bonds

        atom_features = [
            atoms.AtomicElement(),
            atoms.AtomConnectivity(),
            atoms.AtomAverageFormalCharge(),
            atoms.AtomHybridization(),
            atoms.AtomInRingOfSize(ring_size=3),
            atoms.AtomInRingOfSize(ring_size=4),
            atoms.AtomInRingOfSize(ring_size=5),
            atoms.AtomInRingOfSize(ring_size=6),
        ]

        bond_features = [
            bonds.BondInRingOfSize(ring_size=3),
            bonds.BondInRingOfSize(ring_size=4),
            bonds.BondInRingOfSize(ring_size=5),
            bonds.BondInRingOfSize(ring_size=6),
        ]

        model = GNNModel(
            {   "version": "0.1",
                "atom_features": atom_features,
                "bond_features": bond_features,
                "convolution": {
                    "architecture": "SAGEConv",
                    "layers": [
                        {
                            "hidden_feature_size": 128,
                            "activation_function": "ReLU",
                            "dropout": 0.0,
                            "aggregator_type": "mean",
                        }
                    ] * 3
                },
                "readouts": {
                    "am1bcc-charges": {
                        "pooling": "atoms",
                        "postprocess": "compute_partial_charges",
                        "layers": [
                            {
                                "hidden_feature_size": 128,
                                "activation_function": "ReLU",
                                "dropout": 0.0,
                            }
                        ] * 4
                    }
                }
            }
        )

        assert atom_features == model.config.atom_features
        assert bond_features == model.config.bond_features

    def test_to_nagl(self, am1bcc_model):
        pytest.importorskip("dgl")
        assert am1bcc_model._is_dgl
        nagl_model = am1bcc_model._as_nagl()

        original_state_dict = am1bcc_model.state_dict()
        new_state_dict = nagl_model.state_dict()
        assert original_state_dict.keys() == new_state_dict.keys()
        for key in original_state_dict.keys():
            assert torch.allclose(original_state_dict[key], new_state_dict[key])

    def test_compute_property_dgl(self, am1bcc_model, openff_methane_uncharged, expected_methane_charges):
        pytest.importorskip("dgl")
        charges = am1bcc_model._compute_properties_dgl(openff_methane_uncharged)
        charges = charges["am1bcc_charges"].detach().numpy().flatten()
        assert_allclose(charges, expected_methane_charges, atol=1e-5)

    def test_compute_property_networkx(self, am1bcc_model, openff_methane_uncharged, expected_methane_charges):
        charges = am1bcc_model._compute_properties_nagl(openff_methane_uncharged)
        charges = charges["am1bcc_charges"].detach().numpy().flatten()
        assert_allclose(charges, expected_methane_charges, atol=1e-5)

    def test_compute_property_assumed(self, am1bcc_model, openff_methane_uncharged, expected_methane_charges):
        charges = am1bcc_model.compute_property(openff_methane_uncharged, as_numpy=True)
        assert_allclose(charges, expected_methane_charges, atol=1e-5)
    
    def test_compute_property_specified(self, am1bcc_model, openff_methane_uncharged, expected_methane_charges):
        charges = am1bcc_model.compute_property(
            openff_methane_uncharged,
            as_numpy=True,
            readout_name="am1bcc_charges"
        )
        assert_allclose(charges, expected_methane_charges, atol=1e-5)
    
    def test_compute_properties(self, am1bcc_model, openff_methane_uncharged, expected_methane_charges):
        charges = am1bcc_model.compute_properties(
            openff_methane_uncharged,
            as_numpy=True,
        )
        assert len(charges) == 1
        assert_allclose(charges["am1bcc_charges"], expected_methane_charges, atol=1e-5)

    def test_compute_properties_check_domains(self, am1bcc_model, openff_methane_uncharged):
        am1bcc_model.compute_properties(
            openff_methane_uncharged,
            check_domains=True,
            error_if_unsupported=True,
        )
        
    def test_compute_properties_warning_domains(self, am1bcc_model, openff_methyl_methanoate):
        with pytest.warns(UserWarning):
            am1bcc_model.compute_properties(
                openff_methyl_methanoate,
                check_domains=True,
                error_if_unsupported=False,
            )
    
    def test_compute_properties_error_domains(self, am1bcc_model, openff_methyl_methanoate):
        with pytest.raises(ValueError):
            am1bcc_model.compute_properties(
                openff_methyl_methanoate,
                check_domains=True,
                error_if_unsupported=True,
            )


    def test_load(self, openff_methane_uncharged, expected_methane_charges):
        model = GNNModel.load(EXAMPLE_AM1BCC_MODEL, eval_mode=True)
        assert isinstance(model, GNNModel)

        assert model.config.atom_features == [
            AtomicElement(
                categories=["C", "O", "H", "N", "S", "F", "Br", "Cl", "I", "P"]
            ),
            AtomConnectivity(categories=[1, 2, 3, 4, 5, 6]),
            AtomAverageFormalCharge(),
            AtomInRingOfSize(ring_size=3),
            AtomInRingOfSize(ring_size=4),
            AtomInRingOfSize(ring_size=5),
            AtomInRingOfSize(ring_size=6),
        ]

        charges = model.compute_property(openff_methane_uncharged, as_numpy=True)
        assert_allclose(charges, expected_methane_charges, atol=1e-5)

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
        model = GNNModel.load(EXAMPLE_AM1BCC_MODEL, eval_mode=True)
        testdir = importlib_resources.files("openff.nagl") / "tests"
        path = testdir / "data" / "example_am1bcc_sage_charges" / f"{smiles}.sdf"
        molecule = Molecule.from_file(
            str(path), file_format="sdf", allow_undefined_stereo=True
        )

        desired = molecule.partial_charges.m_as(unit.elementary_charge)
        computed = model.compute_property(molecule, as_numpy=True)

        assert_allclose(computed, desired, atol=1e-5)

    def test_save(self, am1bcc_model, openff_methane_uncharged, tmpdir, expected_methane_charges):
        with tmpdir.as_cwd():
            am1bcc_model.save("model.pt")
            model = GNNModel.load("model.pt", eval_mode=True)
            charges = model.compute_property(openff_methane_uncharged, as_numpy=True)
            assert_allclose(charges, expected_methane_charges, atol=1e-5)
