"Classes for handling featurized molecule data to train GNN models"

from collections import defaultdict
import errno
import os
from typing import (
    TYPE_CHECKING,
    Callable,
    Collection,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    Literal,
)

import torch
import tqdm
import pickle
import functools
import pathlib
import pytorch_lightning as pl
from openff.toolkit.topology import Molecule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from openff.nagl._dgl.batch import DGLMoleculeBatch
from openff.nagl._dgl.molecule import DGLMolecule
from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.utils._utils import as_iterable
from openff.nagl.utils._types import Pathlike, FromYamlMixin
from openff.nagl.storage.record import (
    ChargeMethod,
    MoleculeRecord,
    WibergBondOrderMethod,
)


from .label import EmptyLabeller, LabelFunctionLike, LabelPrecomputedMolecule

if TYPE_CHECKING:
    from openff.nagl.storage._store import MoleculeStore


__all__ = [
    "DGLMoleculeDataset",
    "DGLMoleculeDatasetEntry",
]


OpenFFToDGLConverter = Callable[
    ["Molecule", List[AtomFeature], List[BondFeature]], DGLMolecule
]


class DGLMoleculeDatasetEntry(NamedTuple):
    """A named tuple containing a featurized molecule graph, a tensor of the atom
    features, and a tensor of the molecule label.
    """

    molecule: DGLMolecule
    labels: Dict[str, torch.Tensor]

    @classmethod
    def from_openff(
        cls,
        openff_molecule: "Molecule",
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        label_function: LabelFunctionLike,
        openff_to_dgl_converter: OpenFFToDGLConverter = DGLMolecule.from_openff,
    ):
        labels: Dict[str, torch.Tensor] = label_function(openff_molecule)
        dglmol = openff_to_dgl_converter(
            openff_molecule,
            atom_features,
            bond_features,
        )
        return cls(dglmol, labels)


class DGLMoleculeDataset(Dataset):
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index_or_slice):
        return self.entries[index_or_slice]

    def __init__(self, entries: Tuple[DGLMoleculeDatasetEntry, ...] = tuple()):
        self.entries = list(entries)

    @property
    def n_features(self) -> int:
        """Returns the number of atom features"""
        if not len(self):
            return 0

        return self[0][0].atom_features.shape[1]

    @classmethod
    def from_openff(
        cls,
        molecules: Collection["Molecule"],
        label_function: LabelFunctionLike,
        atom_features: Tuple[AtomFeature] = tuple(),
        bond_features: Tuple[BondFeature] = tuple(),
    ):
        entries = [
            DGLMoleculeDatasetEntry.from_openff(
                mol,
                atom_features,
                bond_features,
                label_function,
            )
            for mol in tqdm.tqdm(molecules, desc="Featurizing molecules")
        ]
        return cls(entries)

    @classmethod
    def from_sdf(
        cls,
        *sdf_files,
        label_function: LabelFunctionLike = EmptyLabeller,
        atom_features: Tuple[AtomFeature] = tuple(),
        bond_features: Tuple[BondFeature] = tuple(),
    ):
        offmols = []
        for file in sdf_files:
            mols = as_iterable(Molecule.from_file(file, file_format="sdf"))
            offmols.extend(mols)

        return cls.from_openff(offmols, label_function, atom_features, bond_features)

    @classmethod
    def from_molecule_stores(
        cls,
        molecule_stores: Tuple["MoleculeStore"],
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        partial_charge_method: Optional["ChargeMethod"] = None,
        bond_order_method: Optional["WibergBondOrderMethod"] = None,
        suppress_toolkit_warnings: bool = True,
    ):
        from openff.nagl.toolkits.openff import capture_toolkit_warnings

        if not partial_charge_method and not bond_order_method:
            raise ValueError(
                "Either partial_charge_method or bond_order_method must be " "provided."
            )
        molecule_stores = as_iterable(molecule_stores)
        molecule_records: List["MoleculeRecord"] = [
            record
            for store in molecule_stores
            for record in store.retrieve(
                partial_charge_methods=partial_charge_method or [],
                bond_order_methods=bond_order_method or [],
            )
        ]

        entries = []
        labeller = LabelPrecomputedMolecule(
            partial_charge_method=partial_charge_method,
            bond_order_method=bond_order_method,
        )
        for record in tqdm.tqdm(molecule_records, desc="featurizing molecules"):
            with capture_toolkit_warnings(run=suppress_toolkit_warnings):
                offmol = record.to_openff(
                    partial_charge_method=partial_charge_method,
                    bond_order_method=bond_order_method,
                )
                entry = DGLMoleculeDatasetEntry.from_openff(
                    offmol,
                    atom_features,
                    bond_features,
                    labeller,
                )
                entries.append(entry)

        return cls(entries)


class DGLMoleculeDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Union[DGLMoleculeDataset, ConcatDataset],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate,
            **kwargs,
        )

    @staticmethod
    def _collate(graph_entries: List[DGLMoleculeDatasetEntry]):
        if isinstance(graph_entries[0], DGLMolecule):
            graph_entries = [graph_entries]

        molecules, labels = zip(*graph_entries)

        batched_molecules = DGLMoleculeBatch.from_dgl_molecules(molecules)
        batched_labels = defaultdict(list)

        for molecule_labels in labels:
            for label_name, label_value in molecule_labels.items():
                batched_labels[label_name].append(label_value.reshape(-1, 1))

        batched_labels = {k: torch.vstack(v) for k, v in batched_labels.items()}

        return batched_molecules, batched_labels


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
        self._check_data_cache()

    @staticmethod
    def _as_path_lists(obj) -> List[pathlib.Path]:
        return [pathlib.Path(path) for path in as_iterable(obj)]

    def _prepare_data_from_paths(
        self,
        paths: List[Pathlike],
    ) -> ConcatDataset:
        from openff.nagl.nn.dataset import DGLMoleculeDataset
        from openff.nagl.storage._store import MoleculeStore

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

    def _check_data_cache(self):
        all_paths = [
            self._training_cache_path,
            self._validation_cache_path,
            self._test_cache_path,
        ]
        existing = [
            path for path in all_paths
            if path and path.is_file()
        ]
        if not self.use_cached_data:
            if len(existing) > 0:
                raise FileExistsError(
                    errno.EEXIST,
                    os.strerror(errno.EEXIST),
                    [path.resolve() for path in existing]
                )

    def _prepare_data(self, data_group: Literal["training", "validation", "test"]):
        input_paths = getattr(self, f"{data_group}_set_paths")
        self.data_cache_directory.mkdir(exist_ok=True, parents=True)

        cache_path = self._get_data_cache_path(data_group)
        if cache_path and cache_path.is_file():
            return

        data = self._prepare_data_from_paths(input_paths)
        with cache_path.open("wb") as f:
            pickle.dump(data, f)

    def _get_data_cache_path(
        self, data_group: Literal["training", "validation", "test"]
    ) -> pathlib.Path:

        from openff.nagl.utils._hash import hash_dict

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
        from openff.nagl.utils._hash import hash_dict
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
        from openff.nagl.nn.dataset import DGLMoleculeDataLoader

        def dataloader(batch_size):
            data = getattr(self, data_name)
            size = batch_size if batch_size else len(data)
            return DGLMoleculeDataLoader(data, batch_size=size)

        return functools.partial(dataloader, batch_size)
