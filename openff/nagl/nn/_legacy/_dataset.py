"Classes for handling featurized molecule data to train GNN models"

from collections import defaultdict
import functools
import glob
import pickle
import typing

import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from openff.toolkit import Molecule

from openff.nagl.config.training import TrainingConfig
from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.molecule._dgl import DGLMolecule, DGLMoleculeBatch, DGLMoleculeOrBatch
from openff.nagl.toolkits.openff import capture_toolkit_warnings
from openff.nagl.utils._parallelization import get_mapper_to_processes

import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np



__all__ = [
    "DGLMoleculeDataset",
    "DGLMoleculeDatasetEntry",
]


class DGLMoleculeDatasetEntry(typing.NamedTuple):
    """A named tuple containing a featurized molecule graph, a tensor of the atom
    features, and a tensor of the molecule label.
    """

    molecule: DGLMolecule
    labels: typing.Dict[str, torch.Tensor]

    @classmethod
    def from_openff(
        cls,
        openff_molecule: Molecule,
        labels: typing.Dict[str, typing.Any],
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
        atom_feature_tensor: typing.Optional[torch.Tensor] = None,
        bond_feature_tensor: typing.Optional[torch.Tensor] = None,
        
    ):
        dglmol = DGLMolecule.from_openff(
            openff_molecule,
            atom_features=atom_features,
            bond_features=bond_features,
            atom_feature_tensor=atom_feature_tensor,
            bond_feature_tensor=bond_feature_tensor,
        )

        labels_ = {}
        for key, value in labels.items():
            if value is not None:
                value = np.asarray(value)
                tensor = torch.from_numpy(value)
                if tensor.dtype == torch.float64:
                    tensor = tensor.float()
                labels_[key] = tensor
        return cls(dglmol, labels_)

    def to(self, device: str):
        return type(self)(
            self.molecule.to(device),
            {k: v.to(device) for k, v in self.labels.items()},
        )

    @classmethod
    def from_mapped_smiles(
        cls,
        mapped_smiles: str,
        labels: typing.Dict[str, typing.Any],
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
        atom_feature_tensor: typing.Optional[torch.Tensor] = None,
        bond_feature_tensor: typing.Optional[torch.Tensor] = None,
    ):
        """
        Create a dataset entry from a mapped SMILES string.
        
        Parameters
        ----------
        mapped_smiles
            The mapped SMILES string.
        labels
            The labels for the dataset entry.
            These will be converted to Pytorch tensors.
        atom_features
            The atom features to use.
            If this is provided, an atom_feature_tensor
            should not be provided as it will be generated
            during featurization.
        bond_features
            The bond features to use.
            If this is provided, a bond_feature_tensor
            should not be provided as it will be generated
            during featurization.
        atom_feature_tensor
            The atom feature tensor to use.
            If this is provided, atom_features should not
            be provided as it will be ignored.
        bond_feature_tensor
            The bond feature tensor to use.
            If this is provided, bond_features should not
            be provided as it will be ignored.
        """
        with capture_toolkit_warnings():
            molecule = Molecule.from_mapped_smiles(
                mapped_smiles,
                allow_undefined_stereo=True,
            )
        return cls.from_openff(
            molecule,
            labels,
            atom_features,
            bond_features,
            atom_feature_tensor,
            bond_feature_tensor,
        )

    @classmethod
    def _from_unfeaturized_pyarrow_row(
        cls,
        row: typing.Dict[str, typing.Any],
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
        smiles_column: str = "mapped_smiles",
    ):
        labels = dict(row)
        mapped_smiles = labels.pop(smiles_column)
        return cls.from_mapped_smiles(
            mapped_smiles,
            labels,
            atom_features,
            bond_features,
        )
    
    @classmethod
    def _from_featurized_pyarrow_row(
        cls,
        row: typing.Dict[str, typing.Any],
        atom_feature_column: str,
        bond_feature_column: str,
        smiles_column: str = "mapped_smiles",
    ):
        labels = dict(row)
        mapped_smiles = labels.pop(smiles_column)
        atom_features = labels.pop(atom_feature_column)
        bond_features = labels.pop(bond_feature_column)

        with capture_toolkit_warnings():
            molecule = Molecule.from_mapped_smiles(
                mapped_smiles,
                allow_undefined_stereo=True,
            )

        if atom_features is not None:
            atom_features = torch.tensor(atom_features).float()
            atom_features = atom_features.reshape(len(molecule.atoms), -1)
        
        if bond_features is not None:
            bond_features = torch.tensor(bond_features).float()
            bond_features = bond_features.reshape(len(molecule.bonds), -1)


        return cls.from_mapped_smiles(
            mapped_smiles,
            labels,
            atom_features=[],
            bond_features=[],
            atom_feature_tensor=atom_features,
            bond_feature_tensor=bond_features,
        )

class LazyFeaturizedDGLMoleculeDataset(Dataset):

    def __len__(self):
        return self.n_entries

    def __getitem__(self, index):
        rows = self.table.slice(index, length=1).to_pylist()
        return DGLMoleculeDatasetEntry._from_featurized_pyarrow_row(
            rows[0],
            atom_feature_column="atom_features",
            bond_feature_column="bond_features",
            smiles_column="mapped_smiles",
        )

    def __init__(self, source):
        self.source = source

        with pa.memory_map(source, "rb") as src:
            reader = pa.ipc.open_file(src)
            table = reader.read_all()
        self.table = table
        self.n_entries = self.table.num_rows


class LazyCachedFeaturizedDGLMoleculeDataset(Dataset):

    def __len__(self):
        return self.n_entries

    def __getitem__(self, index):
        filename = f"{self.prefix}_{index}.pkl"
        with open(filename, "rb") as f:
            entry = pickle.load(f)
        return entry
    
    def __init__(
        self,
        prefix: str,
        n_entries: int
    ):
        self.prefix = str(prefix)
        self.n_entries = n_entries

    
    @classmethod
    def from_unfeaturized_pyarrow(
        cls,
        source: pathlib.Path,
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
        smiles_column: str = "mapped_smiles",
        verbose: bool = False,
        n_processes: int = 0,
        columns=None,
        cache_directory=None,
    ):
        from openff.nagl.training.training import DataHash
        if columns is None:
            columns = []
        if smiles_column not in columns:
            columns = [smiles_column] + columns

        if cache_directory is None:
            cache_directory = "."
        else:
            cache_directory = pathlib.Path(cache_directory)
        
        hashed_file = DataHash.from_file(
            source,
            columns=sorted(columns),
            atom_features=atom_features,
            bond_features=bond_features,
        ).to_hash()
        prefix = str((cache_directory / hashed_file).resolve())
        glob_pattern = f"{prefix}_*.pkl"
        matches = list(glob.glob(glob_pattern))

        with pa.memory_map(source, "rb") as src:
            reader = pa.ipc.open_file(src)
            table = reader.read_all(columns=columns)
        n_rows = table.num_rows
        if len(matches) != n_rows:
            raise ValueError(
                "Could not find correct number of featurized entries. "
                f"Expected {n_rows}, found {len(matches)}."
            )
        return cls(prefix, n_rows)

        
class LazyCachedFeaturizedDGLMoleculeDataset2(Dataset):

    def __len__(self):
        return self.n_entries

    def __getitem__(self, index):
        row = self.table.slice(index, length=1).to_pylist()[0]
        data = next(iter(row.values()))
        entry = pickle.loads(data)
        return entry
    
    def __init__(
        self,
        source: str,
    ):
        self.source = str(source)
        with pa.memory_map(source, "rb") as src:
            reader = pa.ipc.open_file(src)
            self.table = reader.read_all()
        self.n_entries = self.table.num_rows

    
    @classmethod
    def from_unfeaturized_pyarrow(
        cls,
        source: pathlib.Path,
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
        smiles_column: str = "mapped_smiles",
        verbose: bool = False,
        n_processes: int = 0,
        columns=None,
        cache_directory=None,
    ):
        from openff.nagl.training.training import DataHash
        if columns is None:
            columns = []
        if smiles_column not in columns:
            columns = [smiles_column] + columns

        if cache_directory is None:
            cache_directory = "."
        else:
            cache_directory = pathlib.Path(cache_directory)
        
        hashed_file = DataHash.from_file(
            source,
            columns=sorted(columns),
            atom_features=atom_features,
            bond_features=bond_features,
        ).to_hash()
        prefix = str((cache_directory / hashed_file).resolve())
        filename = f"{prefix}.arrow"

        return cls(filename)


        





class DGLMoleculeDataset(Dataset):
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index_or_slice):
        return self.entries[index_or_slice]

    def __init__(self, entries: typing.Tuple[DGLMoleculeDatasetEntry, ...] = tuple()):
        self.entries = list(entries)

    @property
    def n_atom_features(self) -> int:
        if not len(self):
            return 0

        return self[0].molecule.atom_features.shape[1]

    @classmethod
    def from_openff(
        cls,
        molecules: typing.Iterable[Molecule],
        atom_features: typing.Optional[typing.List[AtomFeature]] = None,
        bond_features: typing.Optional[typing.List[BondFeature]] = None,
        atom_feature_tensors: typing.Optional[typing.List[torch.Tensor]] = None,
        bond_feature_tensors: typing.Optional[typing.List[torch.Tensor]] = None,
        labels: typing.Optional[typing.List[typing.Dict[str, typing.Any]]] = None,
        label_function: typing.Optional[
            typing.Callable[[Molecule], typing.Dict[str, typing.Any]]
        ] = None,
    ):
        if labels is None:
            labels = [{} for _ in molecules]
        else:
            labels = [dict(label) for label in labels]
        if len(labels) != len(molecules):
            raise ValueError(
                f"The number of labels ({len(labels)}) must match the number of "
                f"molecules ({len(molecules)})."
            )
        if atom_feature_tensors is not None:
            if len(atom_feature_tensors) != len(molecules):
                raise ValueError(
                    f"The number of atom feature tensors ({len(atom_feature_tensors)}) "
                    f"must match the number of molecules ({len(molecules)})."
                )
        else:
            atom_feature_tensors = [None] * len(molecules)

        if bond_feature_tensors is not None:
            if len(bond_feature_tensors) != len(molecules):
                raise ValueError(
                    f"The number of bond feature tensors ({len(bond_feature_tensors)}) "
                    f"must match the number of molecules ({len(molecules)})."
                )
        else:
            bond_feature_tensors = [None] * len(molecules)

        if label_function is not None:
            for molecule, label in zip(molecules, labels):
                label.update(label_function(molecule))

        entries = []
        for molecule, atom_tensor, bond_tensor, label in zip(
            molecules, atom_feature_tensors, bond_feature_tensors, labels
        ):
            entry = DGLMoleculeDatasetEntry.from_openff(
                molecule,
                label,
                atom_features=atom_features,
                bond_features=bond_features,
                atom_feature_tensor=atom_tensor,
                bond_feature_tensor=bond_tensor,
            )
            entries.append(entry)
        
        return cls(entries)
        

    @classmethod
    def from_unfeaturized_pyarrow(
        cls,
        table: pa.Table,
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
        smiles_column: str = "mapped_smiles",
        verbose: bool = False,
        n_processes: int = 0,
    ):
        entry_labels = table.to_pylist()

        if verbose:
            entry_labels = tqdm.tqdm(entry_labels, desc="Featurizing entries")
        
        featurizer = functools.partial(
            DGLMoleculeDatasetEntry._from_unfeaturized_pyarrow_row,
            atom_features=atom_features,
            bond_features=bond_features,
            smiles_column=smiles_column,
        )
        with get_mapper_to_processes(n_processes=n_processes) as mapper:
            entries = list(mapper(featurizer, entry_labels))
        return cls(entries)
    
    @classmethod
    def from_featurized_pyarrow(
        cls,
        table: pa.Table,
        atom_feature_column: str = "atom_features",
        bond_feature_column: str = "bond_features",
        smiles_column: str = "mapped_smiles",
        verbose: bool = False,
        n_processes: int = 0,
    ):
        entry_labels = table.to_pylist()

        if verbose:
            entry_labels = tqdm.tqdm(entry_labels, desc="Featurizing entries")
        
        featurizer = functools.partial(
            DGLMoleculeDatasetEntry._from_featurized_pyarrow_row,
            atom_feature_column=atom_feature_column,
            bond_feature_column=bond_feature_column,
            smiles_column=smiles_column,
        )
        with get_mapper_to_processes(n_processes=n_processes) as mapper:
            entries = list(mapper(featurizer, entry_labels))
        return cls(entries)
        
        
    @classmethod
    def from_unfeaturized_parquet(
        cls,
        paths: typing.Union[pathlib.Path, typing.Iterable[pathlib.Path]],
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
        columns: typing.Optional[typing.List[str]] = None,
        smiles_column: str = "mapped_smiles",
        verbose: bool = False,
        n_processes: int = 0,
    ):
        from openff.nagl.utils._utils import as_iterable

        paths = as_iterable(paths)
        paths = [pathlib.Path(path) for path in paths]

        if columns is not None:
            columns = list(as_iterable(columns))
            if smiles_column not in columns:
                columns = [smiles_column] + columns

        table = pq.read_table(paths, columns=columns)
        return cls.from_unfeaturized_pyarrow(
            table,
            atom_features,
            bond_features,
            smiles_column=smiles_column,
            verbose=verbose,
            n_processes=n_processes,
        )
    

    @classmethod
    def from_featurized_parquet(
        cls,
        paths: typing.Union[pathlib.Path, typing.Iterable[pathlib.Path]],
        atom_feature_column: str = "atom_features",
        bond_feature_column: str = "bond_features",
        columns: typing.Optional[typing.List[str]] = None,
        smiles_column: str = "mapped_smiles",
        verbose: bool = False,
        n_processes: int = 0,
    ):
        from openff.nagl.utils._utils import as_iterable

        paths = as_iterable(paths)
        paths = [pathlib.Path(path) for path in paths]

        if columns is not None:
            columns = list(as_iterable(columns))
            required = [smiles_column, atom_feature_column, bond_feature_column]
            columns = [x for x in columns if x not in required]
            columns = required + columns

        table = pq.read_table(paths, columns=columns)
        return cls.from_featurized_pyarrow(
            table,
            atom_feature_column=atom_feature_column,
            bond_feature_column=bond_feature_column,
            smiles_column=smiles_column,
            verbose=verbose,
            n_processes=n_processes,
        )
    
    def to_pyarrow(
        self,
        atom_feature_column: str = "atom_features",
        bond_feature_column: str = "bond_features",
        smiles_column: str = "mapped_smiles",
    ):
        """
        Convert the dataset to a Pyarrow table.

        This will contain at minimum the smiles, atom features,
        and bond features, using the column names specified as
        arguments. It will also contain any labels that in the entry.
        
        Parameters
        ----------
        atom_feature_column
            The name of the column to use for the atom features.
        bond_feature_column
            The name of the column to use for the bond features.
        smiles_column
            The name of the column to use for the SMILES strings.
        
        
        Returns
        -------
        table
        """
        required_columns = [smiles_column, atom_feature_column, bond_feature_column]
        label_columns = []
        if len(self):
            first_labels = self.entries[0].labels
            label_columns = list(first_labels.keys())

        label_set = set(label_columns)
        
        rows = []
        for dglmol, labels in self.entries:
            atom_features = None
            bond_features = None
            if dglmol.atom_features is not None:
                atom_features = dglmol.atom_features.detach().numpy()
                atom_features = atom_features.astype(float).flatten()
            if dglmol.bond_features is not None:
                bond_features = dglmol.bond_features.detach().numpy()
                bond_features = bond_features.astype(float).flatten()
            
            mol_label_set = set(labels.keys())
            if label_set != mol_label_set:
                raise ValueError(
                    f"The label sets are not consistent. "
                    f"Expected {label_set}, got {mol_label_set}."
                )

            row = [dglmol.mapped_smiles, atom_features, bond_features]
            for label in label_columns:
                row.append(labels[label].detach().numpy().tolist())
            
            rows.append(row)
        
        table = pa.table(
            [*zip(*rows)],
            names=required_columns + label_columns,
        )
        return table
    
    def to_parquet(
        self,
        path: pathlib.Path,
        atom_feature_column: str = "atom_features",
        bond_feature_column: str = "bond_features",
        smiles_column: str = "mapped_smiles",
    ):
        table = self.to_pyarrow(
            atom_feature_column=atom_feature_column,
            bond_feature_column=bond_feature_column,
            smiles_column=smiles_column,
        )
        pq.write_table(table, path)


class DGLMoleculeDataLoader(DataLoader):
    def __init__(
        self,
        dataset: typing.Union[DGLMoleculeDataset, LazyCachedFeaturizedDGLMoleculeDataset, ConcatDataset],
        batch_size: typing.Optional[int] = 1,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # otherwise shared memory issues
            collate_fn=self._collate,
            pin_memory=True,
            **kwargs,
        )

    @staticmethod
    def _collate(graph_entries: typing.List[DGLMoleculeDatasetEntry]):
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
