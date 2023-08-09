import abc
import functools
import logging
import pathlib
import tqdm
import typing

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from openff.toolkit import Molecule
from openff.units import unit

from openff.nagl._base.base import ImmutableModel
from openff.nagl.utils._parallelization import get_mapper_to_processes
from openff.nagl.toolkits.openff import capture_toolkit_warnings

logger = logging.getLogger(__name__)

ChargeMethodType = typing.Literal[
    "am1bcc", "am1-mulliken", "gasteiger", "formal_charges",
    "mmff94", "am1bccnosymspt", "am1elf10", "am1bccelf10"
]

def _append_column_to_table(
    table: pa.Table,
    key: typing.Union[pa.Field, str],
    values: typing.Iterable[typing.Any],
    exist_ok: bool = False
):
    if isinstance(key, pa.Field):
        k_name = key.name
    else:
        k_name = key
    if k_name in table.column_names:
        if exist_ok:
            logger.warning(
                f"Column {k_name} already exists in table. "
                "Overwriting."
            )
            table = table.drop_columns(k_name)
        else:
            raise ValueError(f"Column {k_name} already exists in table")

    table = table.append_column(key, [values])
    return table

class _BaseLabel(ImmutableModel, abc.ABC):
    name: typing.Literal[""]
    exist_ok: bool = False
    smiles_column: str = "mapped_smiles"
    verbose: bool = False

    def _append_column(
        self,
        table: pa.Table,
        key: typing.Union[pa.Field, str],
        values: typing.Iterable[typing.Any],
    ) -> pa.Table:
        return _append_column_to_table(
            table,
            key,
            values,
            exist_ok=self.exist_ok,
        )
        

    @abc.abstractmethod
    def apply(
        self,
        table: pa.Table,
        verbose: bool = False,
    ) -> pa.Table:
        raise NotImplementedError()

class LabelConformers(_BaseLabel):
    name: typing.Literal["conformer_generation"] = "conformer_generation"
    conformer_column: str = "conformers"
    n_conformer_column: str = "n_conformers"
    n_conformer_pool: int = 500
    n_conformers: int = 10
    rms_cutoff: float = 0.05

    def apply(
        self,
        table: pa.Table,
        verbose: bool = False,
    ):
        rms_cutoff = self.rms_cutoff
        if not isinstance(rms_cutoff, unit.Quantity):
            rms_cutoff = rms_cutoff * unit.angstrom

        batch_smiles = table.to_pydict()[self.smiles_column]
        if verbose:
            batch_smiles = tqdm.tqdm(
                batch_smiles,
                desc="Generating conformers",
            )

        data = {
            self.conformer_column: [],
            self.n_conformer_column: [],
        }

        with capture_toolkit_warnings():
            for smiles in batch_smiles:
                mol = Molecule.from_mapped_smiles(
                    smiles,
                    allow_undefined_stereo=True
                )
                mol.generate_conformers(
                    n_conformers=self.n_conformer_pool,
                    rms_cutoff=rms_cutoff, 
                )
                mol.apply_elf_conformer_selection(
                    limit=self.n_conformers,
                )
                conformers = np.ravel([
                    conformer.m_as(unit.angstrom)
                    for conformer in mol.conformers
                ])
                data[self.conformer_column].append(conformers)
                data[self.n_conformer_column].append(len(mol.conformers))
        
        conformer_field = pa.field(
            self.conformer_column, pa.list_(pa.float64())
        )
        n_conformer_field = pa.field(
            self.n_conformer_column, pa.int64()
        )

        table = self._append_column(
            table,
            conformer_field,
            data[self.conformer_column],
        )

        table = self._append_column(
            table,
            n_conformer_field,
            data[self.n_conformer_column],
        )
        return table

def apply_labellers(
    table: pa.Table,
    labellers: typing.Iterable[_BaseLabel],
    verbose: bool = False,
):
    labellers = list(labellers)
    for labeller in labellers:
        table = labeller.apply(table, verbose=verbose)
    return table


def apply_labellers_to_batch_file(
    source: pathlib.Path,
    labellers: typing.Iterable[_BaseLabel] = tuple(),
    verbose: bool = False,
):
    if not labellers:
        return
    source = pathlib.Path(source)
    dataset = ds.dataset(source, format="parquet")
    table = dataset.to_table()
    table = apply_labellers(table, labellers, verbose=verbose)
    with source.open("wb") as f:
        pq.write_table(table, f)


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
        labellers: typing.Iterable[_BaseLabel],
        verbose: bool = False,
        n_processes: int = 0,
    ):
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
        
    
    @staticmethod
    def _assign_charges_single(
        row,
        smiles_column: str = "mapped_smiles",
        charge_method: ChargeMethodType = "formal_charges",
        conformer_column: str = "conformers",
        use_existing_conformers: bool = False,
    ):
        with capture_toolkit_warnings():
            mol = Molecule.from_mapped_smiles(
                row[smiles_column],
                allow_undefined_stereo=True
            )
            shape = (-1, mol.n_atoms, 3)
            if use_existing_conformers:
                conformers = np.asarray(row[conformer_column])
                conformers = conformers.reshape(shape) * unit.angstrom

                charges = []
                for conformer in conformers:
                    mol.assign_partial_charges(
                        charge_method,
                        use_conformers=[conformer],
                    )
                    charges.append(
                        mol.partial_charges.m_as(unit.elementary_charge)
                    )
                return np.mean(charges, axis=0)
            else:
                mol.assign_partial_charges(charge_method)
                return mol.partial_charges.m_as(unit.elementary_charge)

    def _generate_charges_single(
        self,
        filename: str,
        charge_method: ChargeMethodType = "formal_charges",
        charge_column: str = "charges",
        conformer_column: str = "conformers",
        use_existing_conformers: bool = False,
        verbose: bool = False,
        exist_ok: bool = False,
    ):
        batch_dataset = ds.dataset(filename)

        columns = [self.smiles_column]
        if use_existing_conformers:
            columns.append(conformer_column)

        table = batch_dataset.to_table(columns=columns)
        rows = table.to_pylist()
        if verbose:
            rows = tqdm.tqdm(
                rows,
                desc=f"Generating {charge_method} charges",
            )

        charge_column = []

        with capture_toolkit_warnings():
            for row in rows:
                charges = self._assign_charges_single(
                    row,
                    smiles_column=self.smiles_column,
                    charge_method=charge_method,
                    conformer_column=conformer_column,
                    use_existing_conformers=use_existing_conformers,
                )
                charge_column.append(charges)

        charge_field = pa.field(charge_column, pa.list_(pa.float32()))
        self._append_columns_to_batch(
            filename,
            {
                charge_field: charge_column,
            },
            exist_ok=exist_ok,
        )

    def generate_charges(
        self,
        charge_method: ChargeMethodType = "formal_charges",
        charge_column: str = "charges",
        conformer_column: str = "conformers",
        use_existing_conformers: bool = False,
        verbose: bool = False,
        exist_ok: bool = False,
    ):
        for filename in self.dataset.files:
            self._generate_charges_single(
                filename,
                charge_method=charge_method,
                charge_column=charge_column,
                conformer_column=conformer_column,
                use_existing_conformers=use_existing_conformers,
                verbose=verbose,
                exist_ok=exist_ok,
            )