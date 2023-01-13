import errno
import functools
import inspect
import os
import inspect
import pathlib
import pickle
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset

from openff.nagl.dgl.batch import DGLMoleculeBatch
from openff.nagl.dgl.molecule import DGLMolecule
from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.nn.modules.core import ConvolutionModule, ReadoutModule
from openff.nagl.storage.record import ChargeMethod, WibergBondOrderMethod
from openff.nagl.utils.types import Pathlike, FromYamlMixin
from openff.nagl.utils.utils import as_iterable

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.nn.functional.mse_loss(pred, target))


class DGLMoleculeLightningModel(pl.LightningModule):
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
        self, molecule: Union[DGLMolecule, DGLMoleculeBatch]
    ) -> Dict[str, torch.Tensor]:

        self.convolution_module(molecule)

        readouts: Dict[str, torch.Tensor] = {
            readout_type: readout_module(molecule)
            for readout_type, readout_module in self.readout_modules.items()
        }
        return readouts

    def _default_step(
        self,
        batch: Tuple[DGLMolecule, Dict[str, torch.Tensor]],
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


class DGLMoleculeLightningDataModule(pl.LightningDataModule, FromYamlMixin):
    """A utility class that makes loading and featurizing train, validation and test
    sets more compact.

    Parameters
    ----------
    atom_features : List[AtomFeature]
        The set of atom features to compute for each molecule
    bond_features : List[BondFeature]
        The set of bond features to compute for each molecule
    partial_charge_method : Optional[ChargeMethod]
        The type of partial charges to include in the training labels
    bond_order_method : Optional[WibergBondOrderMethod]
        The type of bond orders to include in the training labels
    training_set_paths : Union[str, Tuple[str]]
        The path(s) to the training set(s) stored in an SQLite
        database that can be loaded with an
        :class:`~openff.nagl.storage.store.MoleculeStore`.
        If multiple paths are provided, the datasets will be concatenated.
        If no paths are provided, training will not be performed.
    validation_set_paths : Union[str, Tuple[str]]
        The path(s) to the validation set(s) stored in an SQLite
        database that can be loaded with an
        :class:`~openff.nagl.storage.store.MoleculeStore`.
        If multiple paths are provided, the datasets will be concatenated.
        If no paths are provided, validation will not be performed.
    test_set_paths : Union[str, Tuple[str]]
        The path(s) to the test set(s) stored in an SQLite
        database that can be loaded with an
        :class:`~openff.nagl.storage.store.MoleculeStore`.
        If multiple paths are provided, the datasets will be concatenated.
        If no paths are provided, testing will not be performed.
    output_path : str
        The path to pickle the data module in.
    training_batch_size : Optional[int]
        The batch size to use for training.
        If not provided, all data will be in a single batch.
    validation_batch_size : Optional[int]
        The batch size to use for validation.
        If not provided, all data will be in a single batch.
    test_batch_size : Optional[int]
        The batch size to use for testing.
        If not provided, all data will be in a single batch.
    use_cached_data : bool
        Whether to simply load any data module found at
        the ``output_path`` rather re-generating it using the other provided
        arguments. **No validation is done to ensure the loaded data matches
        the input arguments so be extra careful when using this option**.
        If this is false and a file is found at ``output_path`` an exception
        will be raised.
    """

    @property
    def n_atom_features(self) -> Optional[int]:
        return sum(len(feature) for feature in self.atom_features)

    def __init__(
        self,
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        partial_charge_method: Optional[ChargeMethod] = None,
        bond_order_method: Optional[WibergBondOrderMethod] = None,
        training_set_paths: Union[Pathlike, Tuple[Pathlike]] = tuple(),
        validation_set_paths: Union[Pathlike, Tuple[Pathlike]] = tuple(),
        test_set_paths: Union[Pathlike, Tuple[Pathlike]] = tuple(),
        training_batch_size: Optional[int] = None,
        validation_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        data_cache_directory: Pathlike = "data",
        use_cached_data: bool = False,
    ):
        super().__init__()

        if partial_charge_method is not None:
            partial_charge_method = ChargeMethod(partial_charge_method)
        if bond_order_method is not None:
            bond_order_method = WibergBondOrderMethod(bond_order_method)

        self.atom_features = list(atom_features)
        self.bond_features = list(bond_features)
        self.partial_charge_method = partial_charge_method
        self.bond_order_method = bond_order_method
        self.training_set_paths = self._as_path_lists(training_set_paths)
        self.validation_set_paths = self._as_path_lists(validation_set_paths)
        self.test_set_paths = self._as_path_lists(test_set_paths)
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.use_cached_data = use_cached_data
        self.data_cache_directory = pathlib.Path(data_cache_directory)

        if self.training_set_paths:
            self.train_dataloader = self._default_dataloader(
                "_train_data", self.training_batch_size
            )
        if self.validation_set_paths:
            self.val_dataloader = self._default_dataloader(
                "_val_data", self.validation_batch_size
            )
        if self.test_set_paths:
            self.test_dataloader = self._default_dataloader(
                "_test_data", self.test_batch_size
            )

        self._training_cache_path = self._get_data_cache_path("training")
        self._validation_cache_path = self._get_data_cache_path("validation")
        self._test_cache_path = self._get_data_cache_path("test")

    @staticmethod
    def _as_path_lists(obj) -> List[pathlib.Path]:
        return [pathlib.Path(path) for path in as_iterable(obj)]

    def _prepare_data_from_paths(
        self,
        paths: List[Pathlike],
    ) -> ConcatDataset:
        from openff.nagl.nn.data import DGLMoleculeDataset
        from openff.nagl.storage.store import MoleculeStore

        if not paths:
            return

        datasets = [
            DGLMoleculeDataset.from_molecule_stores(
                MoleculeStore(path),
                partial_charge_method=self.partial_charge_method,
                bond_order_method=self.bond_order_method,
                atom_features=self.atom_features,
                bond_features=self.bond_features,
            )
            for path in paths
        ]
        return ConcatDataset(datasets)

    def _prepare_data(self, data_group: Literal["training", "validation", "test"]):
        input_paths = getattr(self, f"{data_group}_set_paths")

        self.data_cache_directory.mkdir(exist_ok=True, parents=True)
        cache_path = self._get_data_cache_path(data_group)
        if cache_path and cache_path.is_file():
            if self.use_cached_data:
                return
            else:
                raise FileExistsError(
                    errno.EEXIST,
                    os.strerror(errno.EEXIST),
                    cache_path.resolve(),
                )

        data = self._prepare_data_from_paths(input_paths)
        with cache_path.open("wb") as f:
            pickle.dump(data, f)

    def _get_data_cache_path(
        self, data_group: Literal["training", "validation", "test"]
    ) -> pathlib.Path:

        from openff.nagl.utils.hash import hash_dict

        input_hash = hash_dict(getattr(self, f"{data_group}_set_paths"))
        cache_file = (
            f"charge-{self.partial_charge_method}"
            f"_bond-{self.bond_order_method}"
            f"_feat-{self.get_feature_hash()}"
            f"_paths-{input_hash}"
            ".pkl"
        )
        cache_path = self.data_cache_directory / cache_file
        return cache_path

    def _load_data_cache(self, data_group: Literal["training", "validation", "test"]):
        path = getattr(self, f"_{data_group}_cache_path")
        with path.open("rb") as f:
            data = pickle.load(f)
        return data

    def get_feature_hash(self):
        from openff.nagl.utils.hash import hash_dict
        atom_features, bond_features = [], []
        for feature in self.atom_features:
            obj = feature.dict()
            obj["FEATURE_NAME"] = feature.feature_name
            atom_features.append(obj)

        for feature in self.bond_features:
            obj = feature.dict()
            obj["FEATURE_NAME"] = feature.feature_name
            bond_features.append(obj)
        return hash_dict([atom_features, bond_features])

    def prepare_data(self):
        """Prepare the data for training, validation, and testing.

        This method will load the data from the provided paths and pickle
        it in the ``output_path``, as it is not recommended not to assign
        state in this step.
        """
        for stage in ["training", "validation", "test"]:
            self._prepare_data(stage)


    def setup(self, stage: Optional[str] = None):
        self._train_data = self._load_data_cache("training")
        self._val_data = self._load_data_cache("validation")
        self._test_data = self._load_data_cache("test")


    def _default_dataloader(self, data_name, batch_size):
        from openff.nagl.nn.data import DGLMoleculeDataLoader

        def dataloader(batch_size):
            data = getattr(self, data_name)
            size = batch_size if batch_size else len(data)
            return DGLMoleculeDataLoader(data, batch_size=size)

        return functools.partial(dataloader, batch_size)
