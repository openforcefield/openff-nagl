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
from openff.nagl.toolkits.openff import capture_toolkit_warnings

ChargeMethodType = typing.Literal[
    "am1bcc", "am1-mulliken", "gasteiger", "formal_charge",
    "mmff94", "am1bccnosymspt", "am1elf10", "am1bccelf10",
    "zeros"
]

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
        from .utils import _append_column_to_table
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
    name: typing.Literal["label_conformers"] = "label_conformers"
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


class LabelCharges(_BaseLabel):
    name: typing.Literal["label_charges"] = "label_charges"
    charge_method: ChargeMethodType = "formal_charge"
    conformer_column: str = "conformers"
    charge_column: str = "charges"
    use_existing_conformers: bool = False

    @staticmethod
    def _assign_charges(
        mapped_smiles: str = None,
        charge_method: ChargeMethodType = "formal_charges",
        conformers: typing.Optional[unit.Quantity] = None,
        use_existing_conformers: bool = False,
    ) -> np.ndarray:
        with capture_toolkit_warnings():
            mol = Molecule.from_mapped_smiles(
                mapped_smiles,
                allow_undefined_stereo=True
            )
            shape = (-1, mol.n_atoms, 3)
            if use_existing_conformers:
                if conformers is None:
                    raise ValueError(
                        "Conformers must be provided "
                        "if `use_existing_conformers` is True"
                    )
                if not isinstance(conformers, unit.Quantity):
                    conformers = np.asarray(conformers) * unit.angstrom
                conformers = conformers.reshape(shape)

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
            
    def apply(
        self,
        table: pa.Table,
        verbose: bool = False,
    ):
        rows = table.to_pylist()
        if verbose:
            rows = tqdm.tqdm(rows, desc="Assigning charges")
        all_charges = []
        for row in rows:
            row_kwargs = {
                "mapped_smiles": row[self.smiles_column],
                "charge_method": self.charge_method,
                "use_existing_conformers": self.use_existing_conformers,
            }
            if self.use_existing_conformers:
                row_kwargs["conformers"] = row[self.conformer_column]

            charges = self._assign_charges(**row_kwargs)
            all_charges.append(charges)

        charge_field = pa.field(self.charge_column, pa.list_(pa.float64()))
        table = self._append_column(table, charge_field, all_charges)
        return table

        


LabellerType = typing.Union[
    LabelConformers,
    LabelCharges,
]

def apply_labellers(
    table: pa.Table,
    labellers: typing.Iterable[LabellerType],
    verbose: bool = False,
):
    labellers = list(labellers)
    for labeller in labellers:
        table = labeller.apply(table, verbose=verbose)
    return table


def apply_labellers_to_batch_file(
    source: pathlib.Path,
    labellers: typing.Iterable[LabellerType] = tuple(),
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

