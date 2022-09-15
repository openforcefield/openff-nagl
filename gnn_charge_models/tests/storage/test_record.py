import pytest

from openff.toolkit.topology.molecule import Molecule as OFFMolecule
from gnn_charge_models.storage.record import (
    ChargeMethod,
    PartialChargeRecord,
    WibergBondOrderMethod,
    MoleculeRecord
)


class TestMoleculeRecord:

    def test_from_openff(self):
        offmol = OFFMolecule.from_smiles("C")
        record = MoleculeRecord.from_openff(
            offmol,
            partial_charge_methods=["am1bcc"],
            bond_order_methods=["am1"],
        )

        assert len(record.conformers) == 1

        conformer = record.conformers[0]
        assert len(conformer.partial_charges) == 1
        assert ChargeMethod.AM1BCC in conformer.partial_charges
        assert len(conformer.bond_orders) == 1
        assert WibergBondOrderMethod.AM1 in conformer.bond_orders

        assert offmol.conformers is None
