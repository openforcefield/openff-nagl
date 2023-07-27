import functools
import pathlib
import tqdm
import typing

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from openff.toolkit import Molecule
from openff.units import unit

from openff.nagl.utils._parallelization import get_mapper_to_processes
from openff.nagl.toolkits.openff import capture_toolkit_warnings
from openff.nagl.label.labels import LabellerType

class LabelledDataset:

    def __init__(
            self,
            source,
            smiles_column: str = "mapped_smiles",
        ):
        self.source = source
        self.smiles_column = smiles_column
        self._reload()

    def _reload(self):
        self.dataset = ds.dataset(self.source, format="parquet")

    @classmethod
    def from_smiles(
        cls,
        dataset_path: pathlib.Path,
        smiles: typing.Iterable[str],
        smiles_column: str = "mapped_smiles",
        validate_smiles: bool = False,
        mapped: bool = False,
        batch_size: int = 500,
        verbose: bool = False
    ):
        loader = functools.partial(
            Molecule.from_smiles,
            allow_undefined_stereo=True
        )
        mapped_loader = functools.partial(
            Molecule.from_mapped_smiles,
            allow_undefined_stereo=True
        )
        if not mapped:
            converter = lambda x: loader(x).to_smiles(mapped=True)
        elif validate_smiles:
            converter = lambda x: mapped_loader(x).to_smiles(mapped=True)
        else:
            converter = lambda x: x

        if verbose:
            smiles = tqdm.tqdm(smiles, ncols=80, desc="Iterating through SMILES")

        field = pa.field(smiles_column, pa.string())
        data = {
            smiles_column: [converter(smi) for smi in smiles]
        }
        table = pa.Table.from_pydict(data, schema=pa.schema([field]))
        ds.write_dataset(
            table,
            base_dir=dataset_path,
            format="parquet",
            max_rows_per_file=batch_size,
            max_rows_per_group=batch_size
        )
        return cls(dataset_path, smiles_column=smiles_column)
        
    def _append_columns(
        self,
        columns: typing.Dict[pa.Field, typing.Iterable[typing.Any]],
        exist_ok: bool = False,
    ):
        from .utils import _append_column_to_table

        n_all_rows = self.dataset.count_rows()
        all_lengths = {k: len(v) for k, v in columns.items()}
        if not all(n_all_rows == v for v in all_lengths.values()):
            raise ValueError(
                "All columns must have the same number of rows "
                "as the dataset. Given columns have lengths: "
                f"{', '.join(f'{k}: {v}' for k, v in all_lengths.items())}"
            )
        for filename in self.dataset.files:
            batch_dataset = ds.dataset(filename)
            n_rows = batch_dataset.count_rows()
            batch_columns = {
                k: v[:n_rows]
                for k, v in columns.items()
            }

            batch_table = batch_dataset.to_table()
            for k, v in batch_columns.items():
                batch_table = _append_column_to_table(
                    batch_table,
                    k,
                    v,
                    exist_ok=exist_ok,
                )
            with open(filename, "wb") as f:
                pq.write_table(batch_table, f)

            columns = {
                k: v[n_rows:]
                for k, v in columns.items()
            }
        assert all(len(v) == 0 for v in columns.values())
        self._reload()
         
    def apply_labellers(
        self,
        labellers: typing.Iterable[LabellerType],
        verbose: bool = False,
        n_processes: int = 0,
    ):
        from .labels import apply_labellers_to_batch_file

        files = self.dataset.files
        label_func = functools.partial(
            apply_labellers_to_batch_file,
            labellers=labellers,
            verbose=verbose,
        )
        with get_mapper_to_processes(n_processes) as mapper:
            results = mapper(label_func, files)
            if verbose:
                results = tqdm.tqdm(
                    results,
                    desc="Applying labellers to batches",
                )
            list(results)
        self._reload()
        