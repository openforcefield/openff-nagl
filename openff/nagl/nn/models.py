import copy
from typing import TYPE_CHECKING, Tuple, Dict, Union, Callable, Literal, Optional

import torch
import pytorch_lightning as pl

from openff.utilities.exceptions import MissingOptionalDependencyError
from openff.nagl.nn._containers import ConvolutionModule, ReadoutModule
from openff.nagl.config.model import ModelConfig
from openff.nagl.config.training import TrainingConfig

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule
    from openff.nagl.features.atoms import AtomFeature
    from openff.nagl.features.bonds import BondFeature
    from openff.nagl.molecule._dgl import DGLMoleculeOrBatch
    from openff.nagl.nn.postprocess import PostprocessLayer
    from openff.nagl.nn.activation import ActivationFunction
    from openff.nagl.nn.gcn._base import BaseGCNStack


class BaseGNNModel(torch.nn.Module):
    def __init__(
        self,
        convolution_module: ConvolutionModule,
        readout_modules: ReadoutModule,
    ):
        super().__init__()
        self.convolution_module = convolution_module
        self.readout_modules = readout_modules

    def forward(
        self, molecule: "DGLMoleculeOrBatch"
    ) -> Dict[str, torch.Tensor]:
        self.convolution_module(molecule)

        readouts: Dict[str, torch.Tensor] = {
            readout_type: readout_module(molecule)
            for readout_type, readout_module in self.readout_modules.items()
        }
        return readouts

class GNNModel(BaseGNNModel):
    def __init__(self, config: ModelConfig):
        if not isinstance(config, ModelConfig):
            config = ModelConfig(**config)

        convolution_module = ConvolutionModule.from_config(
            config.convolution,
            n_input_features=config.n_atom_features,
        )

        readout_modules = {}
        for readout_name, readout_config in config.readouts.items():
            readout_modules[readout_name] = ReadoutModule.from_config(
                readout_config,
                n_input_features=config.convolution.layers[-1].hidden_feature_size,
            )

        super().__init__(
            convolution_module=convolution_module,
            readout_modules=readout_modules,
        )

        self.save_hyperparameters({"config": config.dict()})
        self.config = config
    
    @property
    def _is_dgl(self):
        return self.convolution_module._is_dgl
    
    def _as_nagl(self):
        copied = type(self)(self.config)
        copied.convolution_module = self.convolution_module._as_nagl(copy_weights=True)
        copied.load_state_dict(self.state_dict())
        return copied
    
    def compute_properties(
        self,
        molecule: "Molecule",
        as_numpy: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the trained property for a molecule.

        Parameters
        ----------
        molecule: :class:`~openff.toolkit.topology.Molecule`
            The molecule to compute the property for.
        as_numpy: bool
            Whether to return the result as a numpy array.
            If ``False``, the result will be a ``torch.Tensor``.

        Returns
        -------
        result: Dict[str, torch.Tensor] or Dict[str, numpy.ndarray]
        """
        try:
            values = self._compute_properties_dgl(molecule)
        except (MissingOptionalDependencyError, TypeError):
            values = self._compute_properties_nagl(molecule)
        if as_numpy:
            values = {k: v.detach().numpy().flatten() for k, v in values.items()}
        return values
    
    def compute_property(
        self,
        molecule: "Molecule",
        readout_name: Optional[str] = None,
        as_numpy: bool = True,
    ):
        """
        Compute the trained property for a molecule.

        Parameters
        ----------
        molecule: :class:`~openff.toolkit.topology.Molecule`
            The molecule to compute the property for.
        readout_name: str
            The name of the readout property to return.
            If this is not given and there is only one readout,
            the result of that readout will be returned.
        as_numpy: bool
            Whether to return the result as a numpy array.
            If ``False``, the result will be a ``torch.Tensor``.

        Returns
        -------
        result: torch.Tensor or numpy.ndarray
        """
        properties = self.compute_properties(
            molecule=molecule,
            as_numpy=as_numpy,
        )
        if readout_name is None:
            if len(properties) == 1:
                return next(iter(properties.values()))
            raise ValueError(
                "The readout name must be specified if the model has multiple readouts"
            )
        return properties[readout_name]

    def _compute_properties_nagl(self, molecule: "Molecule") -> "torch.Tensor":
        from openff.nagl.molecule._graph.molecule import GraphMolecule

        nxmol = GraphMolecule.from_openff(
            molecule,
            atom_features=self.atom_features,
            bond_features=self.bond_features,
        )
        model = self
        if self._is_dgl:
            model = self._as_nagl()
        return model.forward(nxmol)

    def _compute_properties_dgl(self, molecule: "Molecule") -> "torch.Tensor":
        from openff.nagl.molecule._dgl.molecule import DGLMolecule

        if not self._is_dgl:
            raise TypeError(
                "This model is not a DGL-based model "
                 "and cannot be used to compute properties with the DGL backend"
            )

        dglmol = DGLMolecule.from_openff(
            molecule,
            atom_features=self.atom_features,
            bond_features=self.bond_features,
        )
        return self.forward(dglmol)
    
    @classmethod
    def load(cls, model: str, eval_mode: bool = True):
        """
        Load a model from a file.

        Parameters
        ----------
        model: str
            The path to the model to load.
            This should be a file containing a dictionary of
            hyperparameters and a state dictionary,
            with the keys "hyperparameters" and "state_dict".
            This can be created using the `save` method.
        eval_mode: bool
            Whether to set the model to evaluation mode.

        Returns
        -------
        model: GNNModel

        Examples
        --------

        >>> model.save("model.pt")
        >>> new_model = GNNModel.load("model.pt")

        Notes
        -----
        This method is not compatible with normal Pytorch
        models saved with ``torch.save``, as it expects
        a dictionary of hyperparameters and a state dictionary.
        """
        model_kwargs = torch.load(str(model))
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

    def save(self, path: str):
        """
        Save this model to a file.

        Parameters
        ----------
        path: str
            The path to save this file to.

        Examples
        --------

        >>> model.save("model.pt")
        >>> new_model = GNNModel.load("model.pt")

        Notes
        -----
        This method writes a dictionary of the hyperparameters and the state dictionary,
        with the keys "hyperparameters" and "state_dict".
        """
        torch.save(
            {
                "hyperparameters": self.hparams,
                "state_dict": self.state_dict(),
            },
            str(path),
        )


class TrainingGNNModel(pl.LightningModule):

    def __init__(self, config: TrainingConfig):
        if not isinstance(config, TrainingConfig):
            config = TrainingConfig(**config)
        
        self.save_hyperparameters({"config": config.dict()})
        self.config = config

        self.model = GNNModel.from_config(config.model)
        self._data_config = {
            "train": self.config.data.training,
            "val": self.config.data.validation,
            "test": self.config.data.test,
        }

    def _default_step(
        self,
        batch: Tuple["DGLMoleculeOrBatch", Dict[str, torch.Tensor]],
        step_type: Literal["train", "val", "test"],
    ) -> torch.Tensor:       
        molecule, labels = batch
        predictions = self.forward(molecule)
        targets = self._data_config[step_type].targets
        
        loss = torch.zeros(1).type_as(next(iter(predictions.values())))
        for target in targets:
            target_loss = target.evaluate_loss(
                molecules=molecule,
                labels=labels,
                predictions=predictions,
                readout_modules=self.model.readout_modules,
            )
            step_name = (
                f"{step_type}/{target.target_label}/{target.name}//"
                f"{target.metric.name}/{target.weight}/{target.denominator}"
            )
            self.log(step_name, target_loss)
            loss += target_loss


        self.log(f"{step_type}/loss", loss)
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
        config = self.config.optimizer
        if config.optimizer.lower() != "adam":
            raise NotImplementedError(
                f"Optimizer {self.config.optimizer.optimizer} not implemented"
            )
        optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        return optimizer
    
    @property
    def _torch_optimizer(self):
        optimizer = self.optimizers()
        return optimizer.optimizer
    
    
