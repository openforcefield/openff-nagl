import abc
from collections import defaultdict
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
from openff.utilities import requires_package

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



class LabelMultipleDipoles(_BaseLabel):
    name: typing.Literal["multiple_dipoles"] = "multiple_dipoles"
    conformer_column: str = "conformers"
    n_conformer_column: str = "n_conformers"
    charge_column: str = "charges"
    dipole_column: str = "dipoles"

    @staticmethod
    def _calculate_dipoles(
        conformers: np.ndarray,
        n_conformers: int,
        charges: np.ndarray,
        flatten: bool = True,
    ) -> np.ndarray:
        conformers = np.asarray(conformers)
        conformers = conformers.reshape((n_conformers, -1, 3))
        n_atoms = conformers.shape[1]
        charges = np.asarray(charges).reshape(-1, n_atoms).mean(axis=0)
        dipoles = np.matmul(charges, conformers)
        if flatten:
            dipoles = dipoles.reshape(-1)
        return dipoles


    def apply(
        self,
        table: pa.Table,
        verbose: bool = False,
    ):
        rows = table.to_pylist()
        if verbose:
            rows = tqdm.tqdm(rows, desc="Calculating dipoles")
        all_dipoles = []
        for row in rows:
            dipoles = self._calculate_dipoles(
                row[self.conformer_column],
                row[self.n_conformer_column],
                row[self.charge_column],
                flatten=True,
            )
            all_dipoles.append(dipoles)

        dipole_field = pa.field(self.dipole_column, pa.list_(pa.float64()))
        table = self._append_column(table, dipole_field, all_dipoles)
        return table


class LabelMultipleESPs(_BaseLabel):
    name: typing.Literal["multiple_esps"] = "multiple_esps"
    conformer_column: str = "conformers"
    n_conformer_column: str = "n_conformers"
    charge_column: str = "charges"
    inverse_distance_matrix_column: str = "grid_inverse_distances"
    grid_length_column: str = "esp_lengths"
    use_existing_inverse_distances: bool = False
    esp_column: str = "esps"

    @staticmethod
    @requires_package("openff.recharge")
    def _calculate_inverse_distance_grid(
        mapped_smiles: str,
        conformers: np.ndarray,
        n_conformers: int,
    ) -> typing.List[np.ndarray]:
        from openff.recharge.grids import GridGenerator, MSKGridSettings
        with capture_toolkit_warnings():
            mol = Molecule.from_mapped_smiles(
                mapped_smiles,
                allow_undefined_stereo=True
            )

        settings = MSKGridSettings()

        conformers = np.asarray(conformers) * unit.angstrom
        conformers = conformers.reshape((n_conformers, -1, 3))

        all_inv_distances = []
        for conf in conformers:
            grid = GridGenerator.generate(mol, conf, settings)
            displacement = grid[:, None, :] - conf[None, :, :]
            distance = (displacement ** 2).sum(axis=-1) ** 0.5
            distance = distance.m_as(unit.bohr)
            inv_distance = 1 / distance
            all_inv_distances.append(inv_distance)
        return all_inv_distances


    @staticmethod
    def _split_inverse_distance_grid(
        grid_lengths: typing.List[int],
        inverse_distance_matrix: typing.List[float],
        charges: typing.List[float],
    ) -> typing.List[np.ndarray]:
        charges = np.asarray(charges).flatten()
        n_atoms = len(charges)
        total_grid_length = sum(grid_lengths)
        n_total_inv_distances = len(inverse_distance_matrix)
        if total_grid_length * n_atoms != n_total_inv_distances:
            raise ValueError(
                f"Grid length ({total_grid_length}) x n_atoms ({n_atoms}) "
                "must equal length of inverse distances "
                f"({n_total_inv_distances})"
            )
        all_inv_distances = []
        inverse_distance_matrix = np.asarray(inverse_distance_matrix)
        shape = (-1, n_atoms)
        inverse_distance_matrix = inverse_distance_matrix.reshape(shape)
        for n_grid in grid_lengths:
            all_inv_distances.append(inverse_distance_matrix[:n_grid])
            inverse_distance_matrix = inverse_distance_matrix[n_grid:]
        return all_inv_distances

    @staticmethod
    def _calculate_esp(
        inv_distances: np.ndarray,
        charges: np.ndarray,
    ) -> np.ndarray:
        charges = np.asarray(charges).flatten()
        n_atoms = len(charges)
        inv_distances = np.asarray(inv_distances).reshape(-1, n_atoms)
        esp = inv_distances @ charges
        return esp.flatten()


    def apply(
        self,
        table: pa.Table,
        verbose: bool = False,
    ):
        rows = table.to_pylist()
        if verbose:
            rows = tqdm.tqdm(rows, desc="Calculating ESPs")
        data = defaultdict(list)
        for row in rows:
            if self.use_existing_inverse_distances:
                all_inv_distances = self._split_inverse_distance_grid(
                    row[self.grid_length_column],
                    row[self.inverse_distance_matrix_column],
                    row[self.charge_column]
                )
            else:
                all_inv_distances = self._calculate_inverse_distance_grid(
                    row[self.smiles_column],
                    row[self.conformer_column],
                    row[self.n_conformer_column],
                )
            esps = []
            for inv_distances in all_inv_distances:
                esp = self._calculate_esp(
                    inv_distances,
                    row[self.charge_column],
                )
                esps.extend(esp)

            flat_inverse_distances = []
            grid_lengths = []
            for inv_distances in all_inv_distances:
                flat_inverse_distances.extend(inv_distances.flatten())
                grid_lengths.append(len(inv_distances))
            data[self.grid_length_column].append(grid_lengths)
            data[self.inverse_distance_matrix_column].append(
                flat_inverse_distances
            )
            data[self.esp_column].append(esps)

        if not self.use_existing_inverse_distances:
            grid_length_field = pa.field(
                self.grid_length_column,
                pa.list_(pa.int64())
            )
            inverse_distance_field = pa.field(
                self.inverse_distance_matrix_column,
                pa.list_(pa.float64())
            )
            table = self._append_column(
                table,
                grid_length_field,
                data[self.grid_length_column]
            )
            table = self._append_column(
                table,
                inverse_distance_field,
                data[self.inverse_distance_matrix_column]
            )

        esp_field = pa.field(
            self.esp_column,
            pa.list_(pa.float64())
        )
        table = self._append_column(
            table,
            esp_field,
            data[self.esp_column]
        )
        return table



LabellerType = typing.Union[
    LabelConformers,
    LabelCharges,
    LabelMultipleDipoles,
    LabelMultipleESPs,
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

