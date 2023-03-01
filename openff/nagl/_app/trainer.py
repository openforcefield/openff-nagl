import os
import pathlib
from typing import Optional, Tuple, List, Any

import pytorch_lightning as pl
import rich
from pydantic import validator
from typing import List, Union
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from rich import pretty
from rich.console import NewLine

from openff.nagl._base import ImmutableModel
from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.nn._models import GNNModel
from openff.nagl.nn.dataset import DGLMoleculeLightningDataModule
from openff.nagl.storage.record import ChargeMethod, WibergBondOrderMethod
from openff.nagl.utils._types import FromYamlMixin

from openff.nagl.nn._models import GNNModel


class Trainer(ImmutableModel, FromYamlMixin):
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
    training_batch_size: Optional[int] = None
    validation_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None
    data_cache_directory: pathlib.Path = "data"
    use_cached_data: bool = False
    n_gpus: int = 0
    n_epochs: int = 100
    seed: Optional[int] = None
    convolution_dropout: Union[List[float], float] = 0.0
    readout_dropout: Union[List[float], float] = 0.0

    _model = None
    _data_module = None
    _trainer = None
    _logger = None
    _console = None

    @validator("training_set_paths", "validation_set_paths", "test_set_paths", pre=True)
    def _validate_paths(cls, v):
        from openff.nagl.utils._utils import as_iterable

        v = as_iterable(v)
        v = [pathlib.Path(x).resolve() for x in v]
        return v

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
        dct["atom_features"] = tuple(
            [
                {f.feature_name: f.dict(exclude={"feature_name"})}
                for f in self.atom_features
            ]
        )

        dct["bond_features"] = tuple(
            [
                {f.feature_name: f.dict(exclude={"feature_name"})}
                for f in self.bond_features
            ]
        )
        new_dict = dict(dct)
        for k, v in dct.items():
            if isinstance(v, pathlib.Path):
                v = str(v.resolve())
            new_dict[k] = v
        return new_dict

    def to_yaml_file(self, path):
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_simple_dict(), f)

    def to_simple_hash(self) -> str:
        from openff.nagl.utils._hash import hash_dict

        return hash_dict(self.to_simple_dict())

    @property
    def n_atom_features(self):
        return sum(len(feature) for feature in self.atom_features)

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
            convolution_dropout=self.convolution_dropout,
            readout_dropout=self.readout_dropout,
        )
        return model

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

    def train(
        self,
        callbacks=(ModelCheckpoint(save_top_k=1, monitor="val_loss"),),
        logger_name: str = "default",
        checkpoint_file: Optional[str] = None,
    ):
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
        # self.callbacks = callbacks_

        self._trainer = pl.Trainer(
            gpus=self.n_gpus,
            min_epochs=self.n_epochs,
            max_epochs=self.n_epochs,
            logger=self._logger,
            callbacks=list(callbacks_),
        )

        if self.seed is not None:
            pl.seed_everything(self.seed)

        self.trainer.fit(
            self.model, datamodule=self.data_module, ckpt_path=checkpoint_file
        )

        if self.test_set_paths:
            self.trainer.test(self.model, self.data_module)
