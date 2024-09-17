import importlib.resources
import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from openff.units import unit
from openff.toolkit.topology import Molecule
from openff.toolkit.utils.toolkits import RDKIT_AVAILABLE

from openff.nagl.nn.gcn._sage import SAGEConvStack
from openff.nagl.nn._containers import ConvolutionModule, ReadoutModule
from openff.nagl.nn._models import BaseGNNModel, GNNModel
from openff.nagl.nn._pooling import PoolAtomFeatures, PoolBondFeatures
from openff.nagl.nn.postprocess import ComputePartialCharges
from openff.nagl.nn._sequential import SequentialLayers
from openff.nagl.domains import ChemicalDomain
from openff.nagl.lookups import AtomPropertiesLookupTable, AtomPropertiesLookupTableEntry
from openff.nagl.tests.data.files import (
    EXAMPLE_AM1BCC_MODEL,
    EXAMPLE_MODEL_RC3
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
        lookup_table = AtomPropertiesLookupTable(
            property_name="am1bcc_charges",
            properties=[
                AtomPropertiesLookupTableEntry(
                    inchi="InChI=1/H2S/h1H2",
                    mapped_smiles="[H:2][S:1][H:3]",
                    property_value=[-0.1, 0.05, 0.05],
                    provenance={"description": "test"},
                )
            ]
        )
        model.lookup_tables = {"am1bcc_charges": lookup_table}

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
        from openff.toolkit import Molecule

        model = GNNModel.load(EXAMPLE_AM1BCC_MODEL, eval_mode=True)
        testdir = importlib.resources.files("openff.nagl") / "tests"
        path = testdir / "data" / "example_am1bcc_sage_charges" / f"{smiles}.sdf"
        molecule = Molecule.from_file(
            str(path), file_format="sdf", allow_undefined_stereo=True
        )

        desired = molecule.partial_charges.m_as(unit.elementary_charge)
        computed = model.compute_property(molecule, as_numpy=True)

        assert_allclose(computed, desired, atol=1e-5)

    def test_forward_unpostprocessed(self):
        dgl = pytest.importorskip("dgl")
        from openff.toolkit import Molecule

        model = GNNModel.load(EXAMPLE_AM1BCC_MODEL, eval_mode=True)
        molecule = Molecule.from_smiles("C")
        nagl_mol = model._convert_to_nagl_molecule(molecule)
        unpostprocessed = model._forward_unpostprocessed(nagl_mol)
        computed = unpostprocessed["am1bcc_charges"].detach().cpu().numpy()
        assert computed.shape == (5, 2)
        expected = np.array([
            [ 0.166862,  5.489722],
            [-0.431665,  5.454424],
            [-0.431665,  5.454424],
            [-0.431665,  5.454424],
            [-0.431665,  5.454424],
        ])
        assert_allclose(computed, expected, atol=1e-5)

    def test_load_model_with_kwargs(self):
        GNNModel.load(
            EXAMPLE_AM1BCC_MODEL,
            eval_mode=True,
            map_location=torch.device('cpu')
        )

    def test_protein_computable(self):
        """
        Test that working with moderately sized protein
        is feasible in time and memory.

        See Issue #101
        """
        from openff.toolkit import Molecule

        model = GNNModel.load(EXAMPLE_AM1BCC_MODEL, eval_mode=True)

        protein = Molecule.from_smiles(
            "CC[C@H](C)[C@H](NC(=O)CNC(=O)CNC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CC(=O)[O-])NC(=O)[C@H](CC(=O)[O-])NC(=O)[C@H](CCCNC(N)=[NH2+])NC(=O)CNC(=O)[C@H](CS)NC(=O)[C@@H](NC(=O)[C@H](CCCNC(N)=[NH2+])NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CC(C)C)NC(=O)CNC(=O)[C@H](CCCNC(N)=[NH2+])NC(=O)[C@H](CCC(=O)[O-])NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](NC(=O)[C@@H]1CCCN1C(=O)[C@H](CC(N)=O)NC(=O)[C@@H](NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](CCC(=O)[O-])NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CCC(=O)[O-])NC(=O)[C@H](C)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](NC(=O)[C@H](CCC(=O)[O-])NC(=O)[C@H](CO)NC(=O)[C@H](C)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)CNC(=O)[C@H](CO)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)CNC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@@H](NC(=O)CNC(=O)[C@H](CC(=O)[O-])NC(=O)[C@H](C)NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](C)NC(=O)[C@@H](NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)CNC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@@H]1CCCN1C(=O)[C@H](CC(=O)[O-])NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](CC(=O)[O-])NC(=O)[C@H](CC(=O)[O-])NC(=O)CNC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CCC(=O)[O-])NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](NC(=O)CNC(=O)CNC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](NC(=O)[C@H](CC(=O)[O-])NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CCC(=O)[O-])NC(=O)[C@H](C)NC(=O)CNC(=O)[C@H](CCCC[NH3+])NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](C)NC(=O)CNC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H]1CCCN1C(=O)CNC(=O)[C@@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@@H]1CCCN1C(=O)[C@H](C)NC(=O)CNC(=O)[C@@H](NC(=O)[C@H](C)NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](CC(=O)[O-])NC(=O)[C@@H]([NH3+])CCSC)C(C)C)C(C)C)[C@@H](C)CC)C(C)C)[C@@H](C)O)[C@@H](C)CC)[C@@H](C)CC)[C@@H](C)CC)[C@@H](C)CC)[C@@H](C)CC)C(C)C)C(C)C)[C@@H](C)CC)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)[C@@H](C)CC)C(C)C)[C@@H](C)CC)[C@@H](C)CC)[C@@H](C)O)[C@@H](C)O)C(C)C)[C@@H](C)O)[C@@H](C)O)[C@@H](C)O)C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@@H](C)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@H](C(=O)N[C@@H](C)C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@H](C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)NCC(=O)N[C@@H](CCC(N)=O)C(=O)NCC(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@H](C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@H](C(=O)N[C@@H](CCSC)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCC(=O)[O-])C(=O)NCC(=O)N[C@H](C(=O)N[C@@H](CC(N)=O)C(=O)N[C@H](C(=O)N[C@@H](CS[C@H]1CC(=O)N(c2ccc3c(c2)C(=O)OC32c3ccc(O)cc3Oc3cc(O)ccc32)C1=O)C(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](C(=O)N[C@@H](CO)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](CCSC)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@H](C(=O)N[C@@H](CO)C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)NCC(=O)NCC(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)N[C@H](C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](CC(C)C)C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](CCCNC(N)=[NH2+])C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@@H](CCC(N)=O)C(=O)NCC(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](C)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](C(=O)N[C@@H](CO)C(=O)NCC(=O)N[C@@H](CC(=O)[O-])C(=O)NCC(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CO)C(=O)N[C@H](C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@@H](C)C(=O)N[C@H](C(=O)N[C@@H](CCC(=O)[O-])C(=O)NCC(=O)N[C@H](C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)C(=O)NCC(=O)N1CCC[C@H]1C(=O)N[C@@H](CC(=O)[O-])C(=O)N1CCC[C@H]1C(=O)N[C@H](C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCNC(N)=[NH2+])C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CC(N)=O)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@H](C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](CCSC)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](C)C(=O)N[C@H](C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@H](C(=O)N[C@@H](CCC(=O)[O-])C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@H](C(=O)N[C@@H](C)C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](CCCC[NH3+])C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N[C@H](C(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)NCC(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@H](C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CC(=O)[O-])C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@@H](CCCC[NH3+])C(=O)NCC(=O)N[C@@H](CC(=O)[O-])C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](CC(C)C)C(=O)N1CCC[C@H]1C(=O)NCC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@H](C(=O)N[C@@H](CCSC)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCC(=O)[O-])C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](CCCC[NH3+])C(=O)NCC(=O)N1CCC[C@H]1C(=O)N[C@@H](CC(=O)[O-])C(=O)NCC(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@H](C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@H](C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CCC(N)=O)C(=O)NC)[C@@H](C)CC)[C@@H](C)O)C(C)C)[C@@H](C)CC)[C@@H](C)O)C(C)C)C(C)C)[C@@H](C)CC)[C@@H](C)O)C(C)C)[C@@H](C)O)[C@@H](C)O)[C@@H](C)O)C(C)C)[C@@H](C)CC)C(C)C)[C@@H](C)CC)C(C)C)[C@@H](C)CC)[C@@H](C)CC)[C@@H](C)O)[C@@H](C)CC)[C@@H](C)CC)C(C)C)[C@@H](C)CC)C(C)C)C(C)C)C(C)C)[C@@H](C)O)C(C)C)[C@@H](C)O)[C@@H](C)O)[C@@H](C)CC)[C@@H](C)CC)C(C)C"
        )
        model.compute_property(protein, as_numpy=True)

    def test_save(self, am1bcc_model, openff_methane_uncharged, tmpdir, expected_methane_charges):
        with tmpdir.as_cwd():
            am1bcc_model.save("model.pt")
            model = GNNModel.load("model.pt", eval_mode=True)
            charges = model.compute_property(openff_methane_uncharged, as_numpy=True)
            assert_allclose(charges, expected_methane_charges, atol=1e-5)


    def test_check_lookup_table(self, am1bcc_model):
        sh2 = Molecule.from_mapped_smiles("[H:1][S:2][H:3]")
        charges = am1bcc_model.compute_property(
            sh2, as_numpy=True, check_lookup_table=True
        )
        assert_allclose(charges, [0.05, -0.1, 0.05])

    def test_check_no_lookup_table(self, am1bcc_model):
        sh2 = Molecule.from_mapped_smiles("[H:1][S:2][H:3]")
        charges = am1bcc_model.compute_property(
            sh2, as_numpy=True, check_lookup_table=False
        )
        assert_allclose(charges, [0.220583, -0.441167,  0.220583], atol=1e-5)

    def test_outside_lookup_table(self, am1bcc_model):
        nh2 = Molecule.from_smiles("N")
        charges =am1bcc_model.compute_property(
            nh2, as_numpy=True, check_lookup_table=True,
        )
        assert_allclose(
            charges,
            [-0.738375,  0.246125,  0.246125,  0.246125],
            atol=1e-5
        )

class TestGNNModelRC3:

    @pytest.fixture()
    def model(self):
        return GNNModel.load(EXAMPLE_MODEL_RC3, eval_mode=True)
    
    def test_contains_lookup_tables(self, model):
        assert "am1bcc_charges" in model.lookup_tables
        assert len(model.lookup_tables) == 1
        assert len(model.lookup_tables["am1bcc_charges"]) == 13944

    @pytest.mark.parametrize("lookup, expected_charges", [
        (True, [-0.10866 ,  0.027165,  0.027165,  0.027165,  0.027165]),
        (False, [-0.159474,  0.039869,  0.039869,  0.039869,  0.039869])
    ])
    def test_compute_property(
        self, model, openff_methane_uncharged, lookup, expected_charges
    ):
        charges = model.compute_property(
            openff_methane_uncharged,
            as_numpy=True,
            check_lookup_table=lookup,

        )
        assert charges.shape == (5,)
        assert charges.dtype == np.float32

        assert_allclose(charges, expected_charges, atol=1e-5)

    
    def test_assign_partial_charges_to_ion(self, model):
        mol = Molecule.from_smiles("[Cl-]")
        assert mol.n_atoms == 1

        charges = model.compute_property(mol, as_numpy=True).flatten()
        assert np.isclose(charges[-1], -1.)
    
    def test_assign_partial_charges_to_hcl_salt(self, model):
        mol = Molecule.from_mapped_smiles("[Cl-:1].[H+:2]")
        assert mol.n_atoms == 2
        
        charges = model.compute_property(mol, as_numpy=True).flatten()
        assert np.isclose(charges[0], -1.)
        assert np.isclose(charges[1], 1.)

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
    @pytest.mark.parameterize(
        "smiles, expected_formal_charges", [
            ("CCCn1cc[n+](C)c1.C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F", [1, -1]),
        ]
    )
    def test_multimolecule_smiles(self, model, smiles, expected_formal_charges):
        from rdkit import Chem

        mol = Molecule.from_smiles(smiles)
        charges = model.compute_property(mol, as_numpy=True)

        # work out which charges belong to which molecule
        rdmol = mol.to_rdkit()
        # assume lowest atoms are in order of left to right
        fragment_indices = sorted(
            Chem.GetMolFrags(rdmol),
            key=lambda x: min(x),
        )
        assert len(fragment_indices) == len(expected_formal_charges)
        for indices, expected_charge in zip(fragment_indices, expected_formal_charges):
            fragment_charges = charges[list(indices)]
            assert np.allclose(fragment_charges, expected_charge)

        

