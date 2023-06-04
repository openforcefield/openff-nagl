from collections import defaultdict
import functools
import glob
import hashlib
import logging
import pathlib
import typing

import torch
import pytorch_lightning as pl

from openff.nagl.config.training import TrainingConfig
from openff.nagl.config.data import DatasetConfig
from openff.nagl.nn._models import GNNModel
from openff.nagl._base.base import ImmutableModel
from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.nn._dataset import DGLMoleculeDataset, LazyCachedFeaturizedDGLMoleculeDataset

if typing.TYPE_CHECKING:
    from openff.nagl.molecule._dgl import DGLMoleculeOrBatch

logger = logging.getLogger(__name__)

def file_digest(file):
    h = hashlib.sha256()
    with open(file, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

class DataHash(ImmutableModel):
    path_hash: str
    columns: typing.List[str]
    atom_features: typing.List[AtomFeature]
    bond_features: typing.List[BondFeature]

    @classmethod
    def from_file(
        cls,
        path: typing.Union[str, pathlib.Path],
        columns: typing.List[str],
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
    ):
        path = pathlib.Path(path)
        # path_hash = file_digest(path)
        path_hash = str(path.resolve())
        return cls(
            # path_hash=path_hash,
            path_hash=path_hash,
            columns=columns,
            atom_features=atom_features,
            bond_features=bond_features,
        )
    
    def to_hash(self):
        json_str = self.json().encode("utf-8")
        hashed = hashlib.sha256(json_str).hexdigest()
        return hashed


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
            self.log(step_name, target_loss)
            loss += target_loss


        self.log(f"{step_type}/loss", loss)
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

    def prepare_data(self):
        for stage, config in self._dataset_configs.items():
            if config is None or not config.sources:
                continue

            cache_dir = pathlib.Path(config.cache_directory)

            columns = config.get_required_target_columns()

            if config.cache_directory is not None and config.use_cached_data:
                cache_dir.mkdir(exist_ok=True, parents=True)
                datasets = []
                for path in config.sources:
                    file_hash = self._hash_file(path, columns)
                    prefix = str((cache_dir / file_hash).resolve())
                    self._prefixes[stage].append(prefix)

                    matches = glob.glob(f"{prefix}*.pkl")
                    n_entries = len(matches)
                    ds = LazyCachedFeaturizedDGLMoleculeDataset(
                        prefix=prefix,
                        n_entries=n_entries,
                    )
                    datasets.append(ds)
                self._datasets[stage] = torch.utils.data.ConcatDataset(datasets)
        

    # def prepare_data(self):
    #     for stage, config in self._dataset_configs.items():
    #         if config is None or not config.sources:

    #             continue

    #         cache_dir = pathlib.Path(config.cache_directory)

    #         columns = config.get_required_target_columns()

    #         if config.cache_directory is not None and config.use_cached_data:
    #             cache_dir.mkdir(exist_ok=True, parents=True)
    #             for path in config.sources:
    #                 file_hash = self._hash_file(path, columns)
    #                 filename = f"{file_hash}.parquet"
    #                 cached_path = cache_dir / filename
    #                 self._file_hashes[stage].append(str(cached_path.resolve()))
    #                 if cached_path.is_file():
    #                     logger.info(f"Found cached data at {cached_path}")
    #                 else:

    #                     ds = DGLMoleculeDataset.from_unfeaturized_parquet(
    #                         [path],
    #                         columns=columns,
    #                         atom_features=self.config.model.atom_features,
    #                         bond_features=self.config.model.bond_features,
    #                         n_processes=self.n_processes,
    #                         verbose=self.verbose,
    #                     )
    #                     ds.to_parquet(cached_path)
    #                     logger.info(f"Wrote cached data to {cached_path}")

    #         else:
    #             ds = DGLMoleculeDataset.from_unfeaturized_parquet(
    #                 config.sources,
    #                 columns=columns,
    #                 atom_features=self.config.model.atom_features,
    #                 bond_features=self.config.model.bond_features,
    #                 n_processes=self.n_processes,
    #                 verbose=self.verbose,
    #             )
    #             self._datasets[stage] = ds


    def setup(self, **kwargs):
        for stage, config in self._dataset_configs.items():
            if stage in self._datasets:
                continue
            if config is None or not config.sources:
                self._datasets[stage] = None
                continue
            ds = DGLMoleculeDataset.from_featurized_parquet(
                self._file_hashes[stage],
                columns=config.get_required_target_columns(),
                n_processes=self.n_processes,
                verbose=self.verbose,
            )
            self._datasets[stage] = ds
            

    
    def _hash_file(
        self,
        path: typing.Union[str, pathlib.Path],
        columns,
    ) -> str:
        dhash = DataHash.from_file(
            path=path,
            columns=columns,
            atom_features=self.config.model.atom_features,
            bond_features=self.config.model.bond_features,
        )
        return dhash.to_hash()