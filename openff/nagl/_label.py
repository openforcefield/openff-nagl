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

from openff.nagl.utils._parallelization import get_mapper_to_processes
from openff.nagl.toolkits.openff import capture_toolkit_warnings

logger = logging.getLogger(__name__)

ChargeMethodType = typing.Literal[
    "am1bcc", "am1-mulliken", "gasteiger", "formal_charges",
    "mmff94", "am1bccnosymspt", "am1elf10", "am1bccelf10"
]



class LabelledDataset:

    def __init__(
            self,
            source,
            smiles_column: str = "mapped_smiles",
        ):
        self.source = source
        self.dataset = ds.dataset(source, format="parquet")
        self.smiles_column = smiles_column

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
        
            

    @staticmethod
    def _append_columns_to_batch(
        filename: pathlib.Path,
        columns: typing.Dict[pa.Field, typing.Iterable[typing.Any]],
        exist_ok: bool = False,
    ):
        dataset = ds.dataset(filename)
        table = dataset.to_table()
        for k, v in columns.items():
            if isinstance(k, pa.Field):
                k_name = k.name
            else:
                k_name = k
            if k_name in table.column_names:
                if exist_ok:
                    logger.warning(
                        f"Column {k_name} already exists in {filename}. "
                        "Overwriting."
                    )
                    table = table.drop_columns(k_name)
                else:
                    raise ValueError(f"Column {k_name} already exists in {filename}")
            table = table.append_column(k, v)
        with open(filename, "wb") as f:
            pq.write_table(table, f)
        
    
    def generate_conformers(
        self,
        conformer_column: str = "conformers",
        n_conformer_column: str = "n_conformers",
        n_conformer_pool: int = 500,
        n_conformers: int = 10,
        rms_cutoff: float = 0.05,
        verbose: bool = False,
        exist_ok: bool = False,
    ):
        if not isinstance(rms_cutoff, unit.Quantity):
            rms_cutoff = rms_cutoff * unit.angstrom

        for filename in self.dataset.files:
            batch_dataset = ds.dataset(filename)

            table = batch_dataset.to_table(columns=[self.smiles_column])
            batch_smiles = table.to_pydict()[self.smiles_column]
            if verbose:
                batch_smiles = tqdm.tqdm(
                    batch_smiles,
                    desc="Generating conformers",
                )

            data = {
                conformer_column: [],
                n_conformer_column: [],
            }

            with capture_toolkit_warnings():
                for smiles in batch_smiles:
                    mol = Molecule.from_mapped_smiles(
                        smiles,
                        allow_undefined_stereo=True
                    )
                    mol.generate_conformers(
                        n_conformers=n_conformer_pool,
                        rms_cutoff=rms_cutoff, 
                    )
                    mol.apply_elf_conformer_selection(
                        limit=n_conformers,
                    )
                    conformers = np.ravel([
                        conformer.m_as(unit.angstrom)
                        for conformer in mol.conformers
                    ])
                    data[conformer_column].append(conformers)
                    data[n_conformer_column].append(len(mol.conformers))
            
            conformer_field = pa.field(conformer_column, pa.list_(pa.float32()))
            n_conformer_field = pa.field(n_conformer_column, pa.int32())

            self._append_columns_to_batch(
                {
                    conformer_field: data[conformer_column],
                    n_conformer_field: data[n_conformer_column],
                },
                exist_ok=exist_ok,
            )

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