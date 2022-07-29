from typing import Tuple, List, Union, Dict, Any

import torch
from openff.toolkit.topology import Molecule as OFFMolecule

from gnn_charge_models.dgl.molecule import DGLMolecule
from gnn_charge_models.features import FeatureArgs

from gnn_charge_models.nn.lightning import DGLMoleculeLightningModel
from gnn_charge_models.nn.module import ConvolutionModule, ReadoutModule
from gnn_charge_models.nn.pooling import PoolAtomFeatures
from gnn_charge_models.nn.sequential import SequentialLayers
from gnn_charge_models.nn.postprocess import ComputePartialCharges


class PartialChargeModelV1(DGLMoleculeLightningModel):
    def __init__(
        self,
        n_gcn_hidden_features: int,
        n_gcn_layers: int,
        n_am1_hidden_features: int,
        n_am1_layers: int,
        learning_rate: float,
        partial_charge_method: str,
        atom_features: Tuple[Union[str, Dict[str, Any]]] = tuple(),
        bond_features: Tuple[Union[str, Dict[str, Any]]] = tuple(),
    ):
        self.n_gcn_hidden_features = n_gcn_hidden_features
        self.n_gcn_layers = n_gcn_layers
        self.n_am1_hidden_features = n_am1_hidden_features
        self.n_am1_layers = n_am1_layers
        self.learning_rate = learning_rate
        self.partial_charge_method = partial_charge_method
        self.readout_name = f"{partial_charge_method}-charges"


        self.atom_features = [
            FeatureArgs.from_input(feature, feature_type="atoms")
            for feature in atom_features
        ]
        self.bond_features = [
            FeatureArgs.from_input(feature, feature_type="bonds")
            for feature in bond_features
        ]

        self.n_atom_features = sum(
            len(feature)
            for feature in self.instantiate_atom_features()
        )

        # build modules
        convolution = ConvolutionModule(
            architecture="SAGEConv",
            n_input_features=self.n_atom_features,
            hidden_feature_sizes=[n_gcn_hidden_features] * n_gcn_layers,
        )

        readout_activation = ["ReLU"] * n_am1_layers + ["Linear"]
        readout_hidden_features = [n_am1_hidden_features] * n_am1_layers + [2]

        readout = ReadoutModule(
            pooling_layer=PoolAtomFeatures(),
            readout_layers=SequentialLayers.with_layers(
                n_input_features=n_gcn_hidden_features,
                hidden_feature_sizes=readout_hidden_features,
                layer_activation_functions=readout_activation,
            ),
            postprocess_layer=ComputePartialCharges(),
        )

        super().__init__(
            convolution_module=convolution,
            readout_modules={self.readout_name: readout},
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()

    def compute_charges(self, molecule: OFFMolecule) -> torch.Tensor:
        dglmol = DGLMolecule.from_openff(
            molecule,
            atom_features=self.instantiate_atom_features(),
            bond_features=self.instantiate_bond_features(),
        )
        return self.forward(dglmol)[self.readout_name]



    def instantiate_atom_features(self):
        return [feature() for feature in self.atom_features]
    
    def instantiate_bond_features(self):
        return [feature() for feature in self.bond_features]
