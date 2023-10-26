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

from openff.units import unit

from openff.nagl._base.base import ImmutableModel
from openff.nagl.utils._parallelization import get_mapper_to_processes

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
        from openff.toolkit import Molecule
        from openff.nagl.toolkits.openff import capture_toolkit_warnings

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

