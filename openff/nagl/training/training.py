from collections import defaultdict
import functools
import glob
import hashlib
import logging
import pathlib
import pickle
import typing

import torch
import pytorch_lightning as pl

from openff.nagl.config.training import TrainingConfig
from openff.nagl.config.data import DatasetConfig
from openff.nagl.nn._models import GNNModel
from openff.nagl._base.base import ImmutableModel
from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.nn._dataset import (
    DGLMoleculeDataset, DataHash,
    _LazyDGLMoleculeDataset
)

if typing.TYPE_CHECKING:
    from openff.nagl.molecule._dgl import DGLMoleculeOrBatch

logger = logging.getLogger(__name__)




class TrainingGNNModel(pl.LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        if not isinstance(config, TrainingConfig):
            config = TrainingConfig(**config)
        
        self.save_hyperparameters({"config": config.dict()})
        self.config = config

        self.model = GNNModel(config.model)
        self._data_config = {
            "train": self.config.data.training,
            "val": self.config.data.validation,
            "test": self.config.data.test,
        }

    def forward(self, molecule: "DGLMoleculeOrBatch") -> typing.Dict[str, torch.Tensor]:
        outputs = self.model.forward(molecule)
        return outputs

    @classmethod
    def from_yaml(cls, filename):
        config = TrainingConfig.from_yaml(filename)
        return cls(config)
    
    def to_yaml(self, filename):
        self.config.to_yaml(filename)

    def _default_step(
        self,
        batch: typing.Tuple["DGLMoleculeOrBatch", typing.Dict[str, torch.Tensor]],
        step_type: typing.Literal["train", "val", "test"],
    ) -> torch.Tensor:       
        molecule, labels = batch
        predictions = self.forward(molecule)
        targets = self._data_config[step_type].targets

        batch_size = self._data_config[step_type].batch_size
        
        loss = torch.zeros(1).type_as(next(iter(predictions.values())))
        for target in targets:
            target_loss = target.evaluate_loss(
                molecules=molecule,
                labels=labels,
                predictions=predictions,
                readout_modules=self.model.readout_modules,
            )
            step_name = (
                f"{step_type}/{target.target_label}/{target.name}/"
                f"{target.metric.name}/{target.weight}/{target.denominator}"
            )
            self.log(step_name, target_loss, batch_size=batch_size)
            loss += target_loss

        self.log(f"{step_type}/loss", loss, batch_size=batch_size)
        return loss
    
    def training_step(self, train_batch, batch_idx):
        loss = self._default_step(train_batch, "train")
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        loss = self._default_step(val_batch, "val")
        return {"validation_loss": loss}

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
    
    
class DGLMoleculeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: TrainingConfig,
        n_processes: int = 0,
        verbose: bool = True,
    ):
        super().__init__()

        if not isinstance(config, TrainingConfig):
            config = TrainingConfig(**config)
        
        self.config = config
        self.n_processes = n_processes
        self.verbose = verbose
        self._dataset_configs = {
            "train": self.config.data.training,
            "val": self.config.data.validation,
            "test": self.config.data.test,
        }
        self._datasets = {}
        self._file_hashes = defaultdict(list)
        self._prefixes = defaultdict(list)

        for stage, config in self._dataset_configs.items():
            if config is None or not config.sources:
                setattr(self, f"{stage}_dataloader", None)
            else:
                self._create_dataloader(stage)

    def _create_dataloader(
        self,
        stage: typing.Literal["train", "val", "test"],
    ):
        from openff.nagl.nn._dataset import DGLMoleculeDataLoader

        config = self._dataset_configs[stage]

        def dataloader():
            data = self._datasets.get(stage)
            batch_size = config.batch_size
            if batch_size is None:
                batch_size = len(data)
            return DGLMoleculeDataLoader(data, batch_size=batch_size)

        setattr(self, f"{stage}_dataloader", dataloader)

    def _get_dgl_molecule_dataset(
        self,
        config,
        cache_dir,
        columns,
    ):
        if config.lazy_loading:
            loader = functools.partial(
                _LazyDGLMoleculeDataset.from_arrow_dataset,
                format="parquet",
                atom_features=self.config.model.atom_features,
                bond_features=self.config.model.bond_features,
                columns=columns,
                cache_directory=cache_dir,
                use_cached_data=config.use_cached_data,
                n_processes=self.n_processes,
            )
        else:
            loader = functools.partial(
                DGLMoleculeDataset.from_arrow_dataset,
                format="parquet",
                atom_features=self.config.model.atom_features,
                bond_features=self.config.model.bond_features,
                columns=columns,
                n_processes=self.n_processes,
            )

        datasets = []
        for path in config.sources:
            ds = loader(path)
            datasets.append(ds)
        dataset = torch.utils.data.ConcatDataset(datasets)
        return dataset

    def prepare_data(self):
        for stage, config in self._dataset_configs.items():
            if config is None or not config.sources:
                continue

            if config.cache_directory is None:
                cache_dir = pathlib.Path(".")
            else:
                cache_dir = pathlib.Path(config.cache_directory)
            columns = config.get_required_target_columns()

            pickle_hash = self._get_hash_file(
                paths=config.sources,
                columns=columns,
                cache_directory=cache_dir,
                extension=".pkl",
            )

            if pickle_hash.exists():
                if not config.use_cached_data:
                    raise ValueError(
                        "Cached data found but use_cached_data is False: "
                        f"{pickle_hash}"
                    )
                else:
                    logger.info(f"Loading cached data from {pickle_hash}")
                    continue

            dataset = self._get_dgl_molecule_dataset(
                config=config,
                cache_dir=cache_dir,
                columns=columns,
            )

            if not config.lazy_loading and config.use_cached_data:
                with open(pickle_hash, "wb") as f:
                    pickle.dump(dataset, f)
                logger.info(f"Saved data to {pickle_hash}")

    def _setup_stage(self, config, stage: str):
        if config is None or not config.sources:
            return None

        cache_dir = config.cache_directory if config.cache_directory else "."
        columns = config.get_required_target_columns()
        if config.use_cached_data or config.lazy_loading:
            pickle_hash = self._get_hash_file(
                paths=config.sources,
                columns=columns,
                cache_directory=cache_dir,
                extension=".pkl"
            )

            if pickle_hash.exists():
                with open(pickle_hash, "rb") as f:
                    ds = pickle.load(f)
                    return ds
        
        dataset = self._get_dgl_molecule_dataset(
            config=config,
            cache_dir=cache_dir,
            columns=columns,
        )
        return dataset

        
    def setup(self, **kwargs):
        for stage, config in self._dataset_configs.items():
            dataset = self._setup_stage(config, stage)
            # if config is None or not config.sources:
            #     self._datasets[stage] = None
            #     continue

            # cache_dir = config.cache_directory if config.cache_directory else "."
            # columns = config.get_required_target_columns()
            # pickle_hash = self._get_hash_file(
            #     paths=config.sources,
            #     columns=columns,
            #     cache_directory=cache_dir,
            # )
            # if not pickle_hash.exists():
            #     raise FileNotFoundError(
            #         f"Data not found for stage {stage}: {pickle_hash}"
            #     )

            # with open(pickle_hash, "rb") as f:
            #     ds = pickle.load(f)
            self._datasets[stage] = dataset
            

    
    def _get_hash_file(
        self,
        paths: typing.Tuple[typing.Union[str, pathlib.Path], ...] = tuple(),
        columns: typing.Tuple[str, ...] = tuple(),
        cache_directory: typing.Union[pathlib.Path, str] = ".",
        extension: str = ""
    ) -> pathlib.Path:
        dhash = DataHash.from_file(
            *paths,
            columns=columns,
            atom_features=self.config.model.atom_features,
            bond_features=self.config.model.bond_features,
        )
        cache_directory = pathlib.Path(cache_directory)

        return cache_directory / f"{dhash.to_hash()}{extension}"