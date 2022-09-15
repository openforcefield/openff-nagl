
import pathlib
import errno
import os
import pickle
from typing import Dict, Union, Tuple, Callable, List, Optional

import torch
from torch.utils.data import ConcatDataset
import pytorch_lightning as pl

from gnn_charge_models.nn.modules.core import ConvolutionModule, ReadoutModule
from gnn_charge_models.dgl.molecule import DGLMolecule
from gnn_charge_models.dgl.batch import DGLMoleculeBatch

from gnn_charge_models.features.atoms import AtomFeature
from gnn_charge_models.features.bonds import BondFeature
from gnn_charge_models.storage.record import ChargeMethod, WibergBondOrderMethod

from gnn_charge_models.utils.utils import as_iterable
from gnn_charge_models.utils.types import Pathlike

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.nn.functional.mse_loss(pred, target))


class DGLMoleculeLightningModel(pl.LightningModule):
    """A model which applies a graph convolutional step followed by multiple (labelled)
    pooling and readout steps.
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
        all_losses = []

        for label_name, label_values in labels.items():
            pred_values = y_pred[label_name]
            all_losses.append(self.loss_function(pred_values, label_values))

        loss = torch.Tensor(all_losses).sum(axis=0)

        self.log(f"{step_type}_loss", loss)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self._default_step(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        return self._default_step(val_batch, "val")

    def test_step(self, test_batch, batch_idx):
        return self._default_step(test_batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class DGLMoleculeLightningDataModule(pl.LightningDataModule):

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
        output_path: Pathlike = "nagl-data-module.pkl",
        training_batch_size: Optional[int] = None,
        validation_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
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
        self.output_path = pathlib.Path(output_path)
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.use_cached_data = use_cached_data

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

    @staticmethod
    def _as_path_lists(obj) -> List[pathlib.Path]:
        return [pathlib.Path(path) for path in as_iterable(obj)]

    def _prepare_data_from_paths(self, paths: List[Pathlike]) -> ConcatDataset:
        from gnn_charge_models.nn.data import DGLMoleculeDataset
        from gnn_charge_models.storage.store import MoleculeStore
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

    def prepare_data(self):
        # die if we don't want to use cached data but found it anyway
        if self.output_path.is_file():
            if not self.use_cached_data:
                raise FileExistsError(
                    errno.EEXIST,
                    os.strerror(errno.EEXIST),
                    self.output_path.resolve(),
                )
            return

        train_data = self._prepare_data_from_paths(self.training_set_paths)
        val_data = self._prepare_data_from_paths(self.validation_set_paths)
        test_data = self._prepare_data_from_paths(self.test_set_paths)

        # not recommended to assign state here
        # so we pickle and read it back later
        with self.output_path.open("wb") as f:
            pickle.dump((train_data, val_data, test_data), f)

    def setup(self, stage: Optional[str] = None):
        with self.output_path.open("rb") as f:
            self._train_data, self._val_data, self._test_data = pickle.load(f)

    def _default_dataloader(self, data_name, batch_size):
        from gnn_charge_models.nn.data import DGLMoleculeDataLoader

        def dataloader():
            data = getattr(self, data_name)
            if batch_size is None:
                batch_size = len(data)
            return DGLMoleculeDataLoader(data, batch_size=batch_size)
        return dataloader
