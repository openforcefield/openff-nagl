"Classes for handling featurized molecule data to train GNN models"

from collections import defaultdict
import functools
import glob
import hashlib
import io
import logging
import pickle
import tempfile
import typing

import tqdm
import torch
from openff.utilities import requires_package
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from openff.nagl._base.base import ImmutableModel
from openff.nagl.config.training import TrainingConfig
from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.molecule._dgl import DGLMolecule, DGLMoleculeBatch, DGLMoleculeOrBatch
from openff.nagl.utils._parallelization import get_mapper_to_processes
from openff.nagl.utils._hash import digest_file

import pathlib
import numpy as np

if typing.TYPE_CHECKING:
    from openff.toolkit import Molecule


__all__ = [
    "DataHash",
    "DGLMoleculeDataset",
    "DGLMoleculeDatasetEntry",
]

logger = logging.getLogger(__name__)


class DataHash(ImmutableModel):
    """A class for computing the hash of a dataset."""
    path_hash: str
    columns: typing.List[str]
    atom_features: typing.List[AtomFeature]
    bond_features: typing.List[BondFeature]

    @classmethod
    def from_file(
        cls,
        *paths: typing.Union[str, pathlib.Path],
        columns: typing.Optional[typing.List[str]] = None,
        atom_features: typing.Optional[typing.List[AtomFeature]] = None,
        bond_features: typing.Optional[typing.List[BondFeature]] = None,
    ):
        path_hash = ""

        for path in paths:
            path = pathlib.Path(path)
            if path.is_dir():
                for file in path.glob("**/*"):
                    if file.is_file():
                        path_hash += digest_file(file)
            elif path.is_file():
                path_hash += digest_file(path)
            else:
                path_hash += str(path.resolve())

        if columns is None:
            columns = []
        columns = sorted(columns)

        if atom_features is None:
            atom_features = []
        if bond_features is None:
            bond_features = []

        return cls(
            path_hash=path_hash,
            columns=columns,
            atom_features=atom_features,
            bond_features=bond_features,
        )
    
    def to_hash(self):
        json_str = self.json().encode("utf-8")
        hashed = hashlib.sha256(json_str).hexdigest()
        return hashed


def _get_hashed_arrow_dataset_path(
    path: pathlib.Path,
    atom_features: typing.Optional[typing.List[AtomFeature]] = None,
    bond_features: typing.Optional[typing.List[BondFeature]] = None,
    columns: typing.Optional[typing.List[str]] = None,
    directory: typing.Optional[pathlib.Path] = None
) -> pathlib.Path:
    hash_value = DataHash.from_file(
        path,
        columns=columns,
        atom_features=atom_features,
        bond_features=bond_features,
    ).to_hash()
    file_path = f"{hash_value}"
    if directory is not None:
        directory = pathlib.Path(directory)
        return directory / file_path
    return pathlib.Path(file_path)



class DGLMoleculeDatasetEntry(typing.NamedTuple):
    """A named tuple containing a featurized molecule graph, a tensor of the atom
    features, and a tensor of the molecule label.
    """

    molecule: DGLMolecule
    labels: typing.Dict[str, torch.Tensor]

    @classmethod
    def from_openff(
        cls,
        openff_molecule: "Molecule",
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
        from openff.toolkit import Molecule

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
        from openff.toolkit import Molecule

        labels = dict(row)
        mapped_smiles = labels.pop(smiles_column)
        atom_features = labels.pop(atom_feature_column)
        bond_features = labels.pop(bond_feature_column)

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


class _LazyDGLMoleculeDataset(Dataset):
    version = 0.1

    @property
    def schema(self):
        import pyarrow as pa

        return pa.schema([pa.field("pickled", pa.binary())])

    def __len__(self):
        return self.n_entries

    def __getitem__(self, index):
        row = self.table.slice(index, length=1).to_pydict()["pickled"][0]
        entry = pickle.loads(row)
        return entry
    
    @requires_package("pyarrow")
    def __init__(
        self,
        source: str,
    ):
        import pyarrow as pa

        self.source = str(source)
        with pa.memory_map(self.source, "rb") as src:
            reader = pa.ipc.open_file(src)
            self.table = reader.read_all()
        self.n_entries = self.table.num_rows
        self.n_atom_features = (
            self[0].molecule.atom_features.shape[1]
            if len(self)
            else 0
        )
    

    @classmethod
    @requires_package("pyarrow")
    def from_arrow_dataset(
        cls,
        path: pathlib.Path,
        format: str = "parquet",
        atom_features: typing.Optional[typing.List[AtomFeature]] = None,
        bond_features: typing.Optional[typing.List[BondFeature]] = None,
        atom_feature_column: typing.Optional[str] = None,
        bond_feature_column: typing.Optional[str] = None,
        smiles_column: str = "mapped_smiles",
        columns: typing.Optional[typing.List[str]] = None,
        cache_directory: typing.Optional[pathlib.Path] = None,
        use_cached_data: bool = True,
        n_processes: int = 0,
    ):
        import pyarrow as pa
        import pyarrow.dataset as ds

        if columns is not None:
            columns = list(columns)
            if smiles_column not in columns:
                columns.append(smiles_column)

        file_path = _get_hashed_arrow_dataset_path(
            path,
            atom_features,
            bond_features,
            columns,
        ).with_suffix(".arrow")

        if cache_directory is not None:
            cache_directory = pathlib.Path(cache_directory)
            output_path = cache_directory / file_path
        else:
            output_path = file_path

        if use_cached_data:
            if output_path.exists():
                return cls(output_path)
            
        else:
            tempdir = tempfile.TemporaryDirectory()
            output_path = pathlib.Path(tempdir.name) / file_path

        logger.info(f"Featurizing dataset to {output_path}")
        

        if atom_feature_column is None and bond_feature_column is None:
        # set featurizer function
            converter = functools.partial(
                cls._pickle_entry_from_unfeaturized_row,
                atom_features=atom_features,
                bond_features=bond_features,
                smiles_column=smiles_column,
            )
        else:
            converter = functools.partial(
                cls._pickle_entry_from_featurized_row,
                atom_feature_column=atom_feature_column,
                bond_feature_column=bond_feature_column,
                smiles_column=smiles_column,
            )
            if columns is not None and atom_feature_column not in columns:
                columns.append(atom_feature_column)
            if columns is not None and bond_feature_column not in columns:
                columns.append(bond_feature_column)

        input_dataset = ds.dataset(path, format=format)

        with pa.OSFile(str(output_path), "wb") as sink:
            with pa.ipc.new_file(sink, cls.schema) as writer:
                input_batches = input_dataset.to_batches(columns=columns)
                for input_batch in input_batches:
                    with get_mapper_to_processes(n_processes=n_processes) as mapper:
                        pickled = list(mapper(converter, input_batch.to_pylist()))

                    output_batch = pa.RecordBatch.from_arrays(
                        [pa.array(pickled)],
                        schema=cls.schema
                    )
                    writer.write_batch(output_batch)
                    
        return cls(output_path)

    @staticmethod
    def _pickle_entry_from_unfeaturized_row(
        row,
        atom_features=None,
        bond_features=None,
        smiles_column="mapped_smiles",
    ):
        entry = DGLMoleculeDatasetEntry._from_unfeaturized_pyarrow_row(
            row,
            atom_features=atom_features,
            bond_features=bond_features,
            smiles_column=smiles_column,
        )
        f = io.BytesIO()
        pickle.dump(entry, f)
        return f.getvalue()

    @staticmethod
    def _pickle_entry_from_featurized_row(
        row,
        atom_feature_column: str = "atom_features",
        bond_feature_column: str = "bond_features",
        smiles_column: str = "mapped_smiles",
    ):
        entry = DGLMoleculeDatasetEntry._from_featurized_pyarrow_row(
            row,
            atom_feature_column=atom_feature_column,
            bond_feature_column=bond_feature_column,
            smiles_column=smiles_column,
        )
        f = io.BytesIO()
        pickle.dump(entry, f)
        return f.getvalue()
        


class DGLMoleculeDataset(Dataset):
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index_or_slice):
        return self.entries[index_or_slice]

    def __init__(self, entries: typing.Tuple[DGLMoleculeDatasetEntry, ...] = tuple()):
        self.entries = entries

    @property
    def n_atom_features(self) -> int:
        if not len(self):
            return 0

        return self[0].molecule.atom_features.shape[1]

    @classmethod
    @requires_package("pyarrow")
    def from_arrow_dataset(
        cls,
        path: pathlib.Path,
        format: str = "parquet",
        atom_features: typing.Optional[typing.List[AtomFeature]] = None,
        bond_features: typing.Optional[typing.List[BondFeature]] = None,
        atom_feature_column: typing.Optional[str] = None,
        bond_feature_column: typing.Optional[str] = None,
        smiles_column: str = "mapped_smiles",
        columns: typing.Optional[typing.List[str]] = None,
        n_processes: int = 0,
    ):
        import pyarrow.dataset as ds

        if columns is not None:
            columns = list(columns)
            if smiles_column not in columns:
                columns.append(smiles_column)

        if atom_feature_column is None and bond_feature_column is None:
            converter = functools.partial(
                DGLMoleculeDatasetEntry._from_unfeaturized_pyarrow_row,
                atom_features=atom_features,
                bond_features=bond_features,
                smiles_column=smiles_column,
            )
        else:
            converter = functools.partial(
                DGLMoleculeDatasetEntry._from_featurized_pyarrow_row,
                atom_feature_column=atom_feature_column,
                bond_feature_column=bond_feature_column,
                smiles_column=smiles_column,
            )
            if columns is not None and atom_feature_column not in columns:
                columns.append(atom_feature_column)
            if columns is not None and bond_feature_column not in columns:
                columns.append(bond_feature_column)

        

        input_dataset = ds.dataset(path, format=format)
        entries = []

        for input_batch in tqdm.tqdm(
            input_dataset.to_batches(columns=columns),
            desc="Featurizing dataset",
        ):
            for row in tqdm.tqdm(input_batch.to_pylist(), desc="Featurizing batch"):
                entries.append(converter(row))
            # with get_mapper_to_processes(n_processes=n_processes) as mapper:
            #     row_entries = list(mapper(converter, input_batch.to_pylist()))
            #     entries.extend(row_entries)
        return cls(entries)
        

    @classmethod
    def from_openff(
        cls,
        molecules: typing.Iterable["Molecule"],
        atom_features: typing.Optional[typing.List[AtomFeature]] = None,
        bond_features: typing.Optional[typing.List[BondFeature]] = None,
        atom_feature_tensors: typing.Optional[typing.List[torch.Tensor]] = None,
        bond_feature_tensors: typing.Optional[typing.List[torch.Tensor]] = None,
        labels: typing.Optional[typing.List[typing.Dict[str, typing.Any]]] = None,
        label_function: typing.Optional[
            typing.Callable[["Molecule"], typing.Dict[str, typing.Any]]
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

        entries = [
            DGLMoleculeDatasetEntry.from_openff(
                molecule,
                label,
                atom_features=atom_features,
                bond_features=bond_features,
                atom_feature_tensor=atom_tensor,
                bond_feature_tensor=bond_tensor,
            )
            for molecule, atom_tensor, bond_tensor, label in zip(
                molecules, atom_feature_tensors, bond_feature_tensors, labels
            )
        ]

        return cls(entries)

    @requires_package("pyarrow")
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
        import pyarrow as pa

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
    




class DGLMoleculeDataLoader(DataLoader):
    def __init__(
        self,
        dataset: typing.Union[DGLMoleculeDataset, _LazyDGLMoleculeDataset, ConcatDataset],
        batch_size: typing.Optional[int] = 1,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self._collate,
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
