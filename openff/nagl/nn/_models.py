import copy
from typing import TYPE_CHECKING, Tuple, Dict, Union, Callable

import torch
import pytorch_lightning as pl

from openff.utilities.exceptions import MissingOptionalDependencyError
from openff.nagl.nn._containers import ConvolutionModule, ReadoutModule

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule
    from openff.nagl.features.atoms import AtomFeature
    from openff.nagl.features.bonds import BondFeature
    from openff.nagl.molecule._dgl.batch import DGLMoleculeBatch
    from openff.nagl.molecule._dgl.molecule import DGLMolecule


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.nn.functional.mse_loss(pred, target))


class BaseGNNModel(pl.LightningModule):
    """A model which applies a graph convolutional step followed by multiple (labelled)
    pooling and readout steps.

    Parameters
    ----------
    convolution_module
        The graph convolutional module.
    readout_modules
        A dictionary of readout modules, keyed by the name of the readout.
    learning_rate
        The learning rate.
    loss_function
        The loss function. This is RMSE by default, but can be any function
        that takes a predicted and target tensor and returns a scalar loss
        in the form of a ``torch.Tensor``.
    """

    def __init__(
        self,
        convolution_module: ConvolutionModule,
        readout_modules: Dict[str, ReadoutModule],
        learning_rate: float,
        loss_function: Callable = rmse_loss,
    ):
        super().__init__()
        self.convolution_module = convolution_module
        self.readout_modules = torch.nn.ModuleDict(readout_modules)
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def forward(
        self, molecule: Union["DGLMolecule", "DGLMoleculeBatch"]
    ) -> Dict[str, torch.Tensor]:
        self.convolution_module(molecule)

        readouts: Dict[str, torch.Tensor] = {
            readout_type: readout_module(molecule)
            for readout_type, readout_module in self.readout_modules.items()
        }
        return readouts

    def _default_step(
        self,
        batch: Tuple["DGLMolecule", Dict[str, torch.Tensor]],
        step_type: str,
    ) -> torch.Tensor:
        molecule, labels = batch
        y_pred = self.forward(molecule)
        loss = torch.zeros(1).type_as(next(iter(y_pred.values())))

        for label_name, label_values in labels.items():
            pred_values = y_pred[label_name]
            label_loss = self.loss_function(pred_values, label_values)
            loss += label_loss
        self.log(f"{step_type}_loss", loss)
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._default_step(train_batch, "train")
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        loss = self._default_step(val_batch, "val")
        return {"val_loss": loss}

    def test_step(self, test_batch, batch_idx):
        loss = self._default_step(test_batch, "test")
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @property
    def _torch_optimizer(self):
        optimizer = self.optimizers()
        return optimizer.optimizer

    @property
    def _is_dgl(self):
        return self.convolution_module._is_dgl

    def _as_nagl(self):
        copied = copy.deepcopy(self)
        if self._is_dgl:
            copied.convolution_module = copied.convolution_module._as_nagl(
                copy_weights=True
            )
        copied.load_state_dict(self.state_dict())
        return copied


class GNNModel(BaseGNNModel):
    @classmethod
    def from_yaml_file(cls, *paths, **kwargs):
        import yaml

        yaml_kwargs = {}
        for path in paths:
            with open(str(path), "r") as f:
                dct = yaml.load(f, Loader=yaml.FullLoader)
                dct = {k.replace("-", "_"): v for k, v in dct.items()}
                yaml_kwargs.update(dct)
        yaml_kwargs.update(kwargs)
        return cls(**yaml_kwargs)

    @property
    def n_atom_features(self) -> int:
        lengths = [len(feature) for feature in self.atom_features]
        n_features = sum(lengths)
        return n_features

    def __init__(
        self,
        convolution_architecture: str,
        n_convolution_hidden_features: int,
        n_convolution_layers: int,
        n_readout_hidden_features: int,
        n_readout_layers: int,
        activation_function: str,
        postprocess_layer: str,
        readout_name: str,
        learning_rate: float,
        atom_features: Tuple["AtomFeature", ...],
        bond_features: Tuple["BondFeature", ...],
        loss_function: Callable = rmse_loss,
        convolution_dropout: float = 0,
        readout_dropout: float = 0,
    ):
        from openff.nagl.features.atoms import AtomFeature
        from openff.nagl.features.bonds import BondFeature
        from openff.nagl.nn.activation import ActivationFunction
        from openff.nagl.nn.gcn import GCNStackMeta
        from openff.nagl.nn._pooling import PoolAtomFeatures
        from openff.nagl.nn.postprocess import PostprocessLayerMeta
        from openff.nagl.nn._sequential import SequentialLayers

        self.readout_name = readout_name

        convolution_architecture = GCNStackMeta._get_class(convolution_architecture)
        postprocess_layer = PostprocessLayerMeta._get_class(postprocess_layer)
        activation_function = ActivationFunction._get_class(activation_function)
        self.atom_features = self._validate_features(atom_features, AtomFeature)
        self.bond_features = self._validate_features(bond_features, BondFeature)

        hidden_conv = [n_convolution_hidden_features] * n_convolution_layers
        convolution_module = ConvolutionModule(
            architecture=convolution_architecture,
            n_input_features=self.n_atom_features,
            hidden_feature_sizes=hidden_conv,
            layer_dropout=convolution_dropout,
        )

        hidden_readout = [n_readout_hidden_features] * n_readout_layers
        hidden_readout.append(postprocess_layer.n_features)
        readout_activation = [activation_function] * n_readout_layers
        readout_activation.append(ActivationFunction.Identity)
        readout_module = ReadoutModule(
            pooling_layer=PoolAtomFeatures(),
            readout_layers=SequentialLayers.with_layers(
                n_input_features=n_convolution_hidden_features,
                hidden_feature_sizes=hidden_readout,
                layer_activation_functions=readout_activation,
                layer_dropout=readout_dropout,
            ),
            postprocess_layer=postprocess_layer(),
        )

        readout_modules = {readout_name: readout_module}

        super().__init__(
            convolution_module=convolution_module,
            readout_modules=torch.nn.ModuleDict(readout_modules),
            learning_rate=learning_rate,
            loss_function=loss_function,
        )
        self.save_hyperparameters()

    def compute_property(
        self, molecule: "Molecule", as_numpy: bool = False
    ) -> "torch.Tensor":
        try:
            values = self._compute_property_dgl(molecule)
        except MissingOptionalDependencyError:
            values = self._compute_property_nagl(molecule)
        if as_numpy:
            values = values.detach().numpy().flatten()
        return values

    def _compute_property_nagl(self, molecule: "Molecule") -> "torch.Tensor":
        from openff.nagl.molecule._graph.molecule import GraphMolecule

        nxmol = GraphMolecule.from_openff(
            molecule,
            atom_features=self.atom_features,
            bond_features=self.bond_features,
        )
        model = self
        if self._is_dgl:
            model = self._as_nagl()
        return model.forward(nxmol)[self.readout_name]

    def _compute_property_dgl(self, molecule: "Molecule") -> "torch.Tensor":
        from openff.nagl.molecule._dgl.molecule import DGLMolecule

        dglmol = DGLMolecule.from_openff(
            molecule,
            atom_features=self.atom_features,
            bond_features=self.bond_features,
        )
        return self.forward(dglmol)[self.readout_name]

    @staticmethod
    def _validate_features(features, feature_class):
        if isinstance(features, dict):
            features = list(features.items())
        all_v = []
        for item in features:
            if isinstance(item, dict):
                all_v.extend(list(item.items()))
            elif isinstance(item, (str, feature_class, type(feature_class))):
                all_v.append((item, {}))
            else:
                all_v.append(item)

        instantiated = []
        for klass, args in all_v:
            if isinstance(klass, feature_class):
                instantiated.append(klass)
            else:
                klass = type(feature_class)._get_class(klass)
                if not isinstance(args, dict):
                    item = klass._with_args(args)
                else:
                    item = klass(**args)
                instantiated.append(item)
        return instantiated

    @classmethod
    def load(cls, model: str, eval_mode: bool = True):
        import torch

        model_kwargs = torch.load(model)
        if isinstance(model_kwargs, dict):
            model = cls(**model_kwargs["hyperparameters"])
            model.load_state_dict(model_kwargs["state_dict"])
        elif isinstance(model_kwargs, cls):
            model = model_kwargs
        else:
            raise ValueError(f"Unknown model type {type(model_kwargs)}")
        if eval_mode:
            model.eval()

        return model
