import pathlib
import os
from typing import Any, Dict, Optional, Tuple, Union
from pydantic import validator

import pytorch_lightning as pl
import rich
import torch
from openff.toolkit.topology.molecule import Molecule as OFFMolecule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from rich import pretty
from rich.console import NewLine

from openff.nagl.base import ImmutableModel
from openff.nagl.dgl.molecule import DGLMolecule
from openff.nagl.features import FeatureArgs, AtomFeature, BondFeature
from openff.nagl.nn.activation import ActivationFunction
from openff.nagl.nn.gcn import GCNStackMeta
from openff.nagl.nn.modules.core import ConvolutionModule, ReadoutModule
from openff.nagl.nn.modules.lightning import (
    DGLMoleculeLightningDataModule,
    DGLMoleculeLightningModel,
)
from openff.nagl.nn.modules.pooling import PoolAtomFeatures
from openff.nagl.nn.modules.postprocess import PostprocessLayerMeta
from openff.nagl.nn.sequential import SequentialLayers
from openff.nagl.storage.record import ChargeMethod, WibergBondOrderMethod

from openff.nagl.nn.models import GNNModel


class Trainer(ImmutableModel):
    convolution_architecture: str
    n_convolution_hidden_features: int
    n_convolution_layers: int
    readout_name: str
    n_readout_hidden_features: int
    n_readout_layers: int
    learning_rate: float
    activation_function: str
    postprocess_layer: str
    atom_features: Tuple[AtomFeature, ...] = tuple()
    bond_features: Tuple[BondFeature, ...] = tuple()
    partial_charge_method: Optional[ChargeMethod] = None
    bond_order_method: Optional[WibergBondOrderMethod] = None
    training_set_paths: Tuple[pathlib.Path, ...] = tuple()
    validation_set_paths: Tuple[pathlib.Path, ...] = tuple()
    test_set_paths: Tuple[pathlib.Path, ...] = tuple()
    output_directory: pathlib.Path = "."
    # output_data_path: pathlib.Path = "nagl-data-module.pkl"
    training_batch_size: Optional[int] = None
    validation_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None
    data_cache_directory: pathlib.Path = "data"
    use_cached_data: bool = False
    n_gpus: int = 0
    n_epochs: int = 100
    seed: Optional[int] = None

    _model = None
    _data_module = None
    _trainer = None
    _logger = None
    _console = None

    @validator("training_set_paths", "validation_set_paths", "test_set_paths", pre=True)
    def _validate_paths(cls, v):
        from openff.nagl.utils.utils import as_iterable
        v = as_iterable(v)
        v = [pathlib.Path(x).resolve() for x in v]
        return v

    # @validator("convolution_architecture", pre=True)
    # def _validate_convolution_architecture(cls, v):
    #     return GCNStackMeta._get_class(v)

    # @validator("postprocess_layer", pre=True)
    # def _validate_postprocess_layer(cls, v):
    #     return PostprocessLayerMeta._get_class(v)
    
    # @validator("activation_function", pre=True)
    # def _validate_activation_function(cls, v):
    #     return ActivationFunction._get_class(v)

    
    @validator("atom_features", "bond_features", pre=True)
    def _validate_atom_features(cls, v, field):
        if isinstance(v, dict):
            v = list(v.items())
        all_v = []
        for item in v:
            if isinstance(item, dict):
                all_v.extend(list(item.items()))
            elif isinstance(item, (str, field.type_, type(field.type_))):
                all_v.append((item, {}))
            else:
                all_v.append(item)

        instantiated = []
        for klass, args in all_v:
            if isinstance(klass, (AtomFeature, BondFeature)):
                instantiated.append(klass)
            else:
                klass = type(field.type_)._get_class(klass)
                if not isinstance(args, dict):
                    item = klass._with_args(args)
                else:
                    item = klass(**args)
                instantiated.append(item)
        return instantiated

    def to_simple_dict(self):
        dct = self.dict()
        dct["atom_features"] = tuple([
            {f.feature_name: f.dict(exclude={"feature_name"})}
            for f in self.atom_features
        ])

        dct["bond_features"] = tuple([
            {f.feature_name: f.dict(exclude={"feature_name"})}
            for f in self.bond_features
        ])
        new_dict = dict(dct)
        for k, v in dct.items():
            if isinstance(v, pathlib.Path):
                v = str(v.resolve())
            new_dict[k] = v
        return new_dict

    # def to_simple_dict(self):
    #     dct = self.dict()
    #     dct["convolution_architecture"] = self.convolution_architecture.name
    #     dct["postprocess_layer"] = self.postprocess_layer.name
    #     dct["activation_function"] = self.activation_function.name
    #     dct["atom_features"] = tuple([
    #         {f.feature_name: f.dict(exclude={"feature_name"})}
    #         for f in self.atom_features
    #     ])

    #     dct["bond_features"] = tuple([
    #         {f.feature_name: f.dict(exclude={"feature_name"})}
    #         for f in self.bond_features
    #     ])
    #     new_dict = dict(dct)
    #     for k, v in dct.items():
    #         if isinstance(v, pathlib.Path):
    #             v = str(v.resolve())
    #         new_dict[k] = v
    #     return new_dict

    def to_yaml_file(self, path):
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.to_simple_dict(), f)

    def to_simple_hash(self) -> str:
        from openff.nagl.utils.hash import hash_dict

        return hash_dict(self.to_simple_dict())

    @classmethod
    def from_yaml_file(cls, *paths, **kwargs):
        import yaml

        yaml_kwargs = {}
        for path in paths:
            with open(str(path), "r") as f:
                dct = yaml.load(f, Loader=yaml.Loader)
                dct = {k.replace("-", "_"): v for k, v in dct.items()}
                yaml_kwargs.update(dct)
        yaml_kwargs.update(kwargs)
        return cls(**yaml_kwargs)


    @property
    def n_atom_features(self):
        return sum(
            len(feature) for feature in self.atom_features
        )

    def _set_up_data_module(self):
        return DGLMoleculeLightningDataModule(
            atom_features=self.atom_features,
            bond_features=self.bond_features,
            partial_charge_method=self.partial_charge_method,
            bond_order_method=self.bond_order_method,
            training_set_paths=self.training_set_paths,
            training_batch_size=self.training_batch_size,
            validation_set_paths=self.validation_set_paths,
            validation_batch_size=self.validation_batch_size,
            test_set_paths=self.test_set_paths,
            test_batch_size=self.test_batch_size,
            use_cached_data=self.use_cached_data,
            # output_path=self.output_data_path,
            data_cache_directory=self.data_cache_directory,
        )

    def _set_up_model(self):
        model = GNNModel(
            convolution_architecture=self.convolution_architecture,
            n_convolution_hidden_features=self.n_convolution_hidden_features,
            n_convolution_layers=self.n_convolution_layers,
            n_readout_hidden_features=self.n_readout_hidden_features,
            n_readout_layers=self.n_readout_layers,
            activation_function=self.activation_function,
            postprocess_layer=self.postprocess_layer,
            readout_name=self.readout_name,
            learning_rate=self.learning_rate,
            atom_features=self.atom_features,
            bond_features=self.bond_features,
        )
        return model

        # convolution_module = self._set_up_convolution_module()
        # readout_module = self._set_up_readout_module()
        # model = DGLMoleculeLightningModel(
        #     convolution_module=convolution_module,
        #     readout_modules={self.readout_name: readout_module},
        #     learning_rate=self.learning_rate,
        # )
        # return model

    # def _set_up_convolution_module(self):
    #     hd = [self.n_convolution_hidden_features] * self.n_convolution_layers
    #     return ConvolutionModule(
    #         architecture=self.convolution_architecture,
    #         n_input_features=self.n_atom_features,
    #         hidden_feature_sizes=hd,
    #     )

    # def _set_up_readout_module(self):
    #     hd = [self.n_readout_hidden_features] * self.n_readout_layers
    #     # TODO: unhardcode this
    #     hd.append(2)
    #     readout_activation = [self.activation_function] * self.n_readout_layers
    #     readout_activation.append(ActivationFunction.Identity)
    #     return ReadoutModule(
    #         pooling_layer=PoolAtomFeatures(),
    #         readout_layers=SequentialLayers.with_layers(
    #             n_input_features=self.n_convolution_hidden_features,
    #             hidden_feature_sizes=hd,
    #             layer_activation_functions=readout_activation,
    #         ),
    #         postprocess_layer=self.postprocess_layer(),
    #     )

    # def compute_property(self, molecule: OFFMolecule) -> torch.Tensor:
    #     dglmol = DGLMolecule.from_openff(
    #         molecule,
    #         atom_features=self.atom_features,
    #         bond_features=self.bond_features,
    #     )
    #     return self.model.forward(dglmol)[self.readout_name]

    def prepare(self):
        self._model = self._set_up_model()
        self._data_module = self._set_up_data_module()

    @property
    def trainer(self):
        return self._trainer
    
    @property
    def logger(self):
        return self._logger

    @property
    def model(self):
        return self._model
    
    @property
    def data_module(self):
        return self._data_module


    def train(self, callbacks=(ModelCheckpoint(save_top_k=1, monitor="val_loss"),), logger_name: str = "default", checkpoint_file: Optional[str] = None):

        if self._model is None:
            self.prepare()

        os.makedirs(str(self.output_directory), exist_ok=True)

        self._console = rich.get_console()
        pretty.install(self._console)

        self._console.print(NewLine())
        self._console.rule("model")
        self._console.print(NewLine())
        self._console.print(self.model.hparams)
        self._console.print(NewLine())
        self._console.print(self.model)
        self._console.print(NewLine())
        self._console.print(NewLine())
        self._console.rule("training")
        self._console.print(NewLine())
        self._console.print(f"Using {self.n_gpus} GPUs")

        self._logger = TensorBoardLogger(
            self.output_directory,
            name=logger_name,
        )

        callbacks_ = [TQDMProgressBar()]
        callbacks_.extend(callbacks)

        self._trainer = pl.Trainer(
            gpus=self.n_gpus,
            min_epochs=self.n_epochs,
            max_epochs=self.n_epochs,
            logger=self._logger,
            callbacks=callbacks_,
        )

        if self.seed is not None:
            pl.seed_everything(self.seed)

        self.trainer.fit(
            self.model,
            datamodule=self.data_module,
            ckpt_path=checkpoint_file
        )

        if self.test_set_paths:
            self.trainer.test(self.model, self.data_module)
