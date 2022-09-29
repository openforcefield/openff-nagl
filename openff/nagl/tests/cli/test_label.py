import pytest
import os

from click.testing import CliRunner
import numpy as np

from openff.nagl.cli.label import label_molecules_cli
from openff.nagl.storage.store import MoleculeStore
from openff.nagl.storage.record import WibergBondOrder

def test_label_molecule(
    openff_methane_uncharged,
):
    runner = CliRunner()
    with runner.isolated_filesystem():
        db_file = "methane.sqlite"

        assert not os.path.isfile(db_file)

        openff_methane_uncharged.to_file("methane.sdf", "sdf")

        label_arguments = [
            "--input-file",
            "methane.sdf",
            "--output-file",
            db_file,
            "--partial-charge-method",
            "am1bcc",
            "--partial-charge-method",
            "am1",
            "--bond-order-method",
            "am1",
        ]

        result = runner.invoke(label_molecules_cli, label_arguments)
        if result.exit_code != 0:
            raise result.exception
        
        assert os.path.isfile(db_file)
        
        store = MoleculeStore(db_file)
        assert len(store) == 1
        record = store.retrieve()[0]

        assert len(record.conformers) == 1
        conformer = record.conformers[0]
        
        assert len(conformer.partial_charges) == 2
        assert set(conformer.partial_charges.keys()) == {"am1bcc", "am1"}
        assert len(conformer.bond_orders) == 1
        assert set(conformer.bond_orders.keys()) == {"am1"}

        for charges in conformer.partial_charges.values():
            assert not np.allclose(charges.values, 0.0)
        for bonds in conformer.bond_orders.values():
            orders = [bond.bond_order for bond in bonds.values]
            assert not np.allclose(orders, 0.0)

