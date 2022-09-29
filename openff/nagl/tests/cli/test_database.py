import os
import pytest

from click.testing import CliRunner
from numpy.testing import assert_allclose

from openff.toolkit.topology import Molecule
from openff.nagl.cli.database import database_cli
from openff.nagl.storage.store import MoleculeStore
from openff.nagl.storage.record import MoleculeRecord
from openff.nagl.utils.openff import get_unitless_charge, stream_molecules_from_file


def test_store_molecule(
    openff_methane_charged,
    openff_methane_charges,
):
    runner = CliRunner()
    with runner.isolated_filesystem():

        openff_methane_charged.to_file("methane.sdf", "sdf")

        new_molecule = list(stream_molecules_from_file("methane.sdf"))[0]
        new_charges = [get_unitless_charge(x) for x in new_molecule.partial_charges]
        assert_allclose(new_charges, openff_methane_charges)


        db_file = "methane.sqlite"

        store_arguments = [
            "store-molecules",
            "--input-file",
            "methane.sdf",
            "--output-file",
            db_file,
            "--partial-charge-method",
            "am1bcc",
        ]

        assert not os.path.isfile(db_file)

        result = runner.invoke(database_cli, store_arguments)
        if result.exit_code:
            raise result.exception
        
        assert os.path.isfile(db_file)

        log_file = "methane-errors.log"
        assert os.path.isfile(log_file)
        with open(log_file, "r") as f:
            contents = f.read()
        assert not contents

        store = MoleculeStore(db_file)
        assert len(store) == 1
        record = store.retrieve()[0]
        assert record.mapped_smiles == openff_methane_charged.to_smiles(mapped=True)

        assert len(record.conformers) == 1
        conformer = record.conformers[0]

        assert len(conformer.partial_charges) == 1
        assert len(conformer.bond_orders) == 0

        values = conformer.partial_charges["am1bcc"].values
        assert_allclose(values, openff_methane_charges)


def test_retrieve_molecule(
    openff_methane_charged,
    openff_methane_charges,
):
    runner = CliRunner()
    with runner.isolated_filesystem():
        db_file = "methane.sqlite"
        sdf_file = "custom_methane.sdf"

        assert not os.path.isfile(db_file)
        assert not os.path.isfile(sdf_file)

        store = MoleculeStore(db_file)
        record = MoleculeRecord.from_precomputed_openff(
            openff_methane_charged,
            partial_charge_method="custom",
        )
        store.store(records=[record])

        assert len(store) == 1

        retrieve_arguments = [
            "retrieve-molecules",
            "--input-file",
            db_file,
            "--output-file",
            sdf_file,
            "--partial-charge-method",
            "custom",
        ]

        result = runner.invoke(database_cli, retrieve_arguments)
        if result.exit_code:
            raise result.exception
        
        assert os.path.isfile(sdf_file)

        from_db_offmol = Molecule.from_file(sdf_file)
        from_db_charges = [get_unitless_charge(x) for x in from_db_offmol.partial_charges]
        assert_allclose(from_db_charges, openff_methane_charges)
