import re

import numpy as np
import pytest

from openff.nagl.storage.db import DB_VERSION, DBInformation
from openff.nagl.storage.session import IncompatibleDBVersion
from openff.nagl.storage.store import (
    ChargeMethod,
    ConformerRecord,
    MoleculeRecord,
    MoleculeStore,
    PartialChargeRecord,
    WibergBondOrderMethod,
    WibergBondOrderRecord,
)


@pytest.fixture()
def tmp_molecule_store(tmp_path) -> MoleculeStore:

    store = MoleculeStore(f"{tmp_path}.sqlite")
    argon = MoleculeRecord(
        mapped_smiles="[Ar:1]",
        conformers=[
            ConformerRecord(
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                partial_charges=[PartialChargeRecord(
                    method="am1", values=[0.5])],
                bond_orders=[],
            )
        ],
    )
    helium = MoleculeRecord(
        mapped_smiles="[He:1]",
        conformers=[
            ConformerRecord(
                coordinates=np.array([[0.0, 0.0, 0.0]]),
                partial_charges=[PartialChargeRecord(
                    method="am1bcc", values=[-0.5])],
                bond_orders=[],
            )
        ],
    )
    chloride = MoleculeRecord(
        mapped_smiles="[Cl:1][Cl:2]",
        conformers=[
            ConformerRecord(
                coordinates=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                partial_charges=[
                    PartialChargeRecord(method="am1", values=[0.5, -0.5]),
                    PartialChargeRecord(method="am1bcc", values=[0.75, -0.75]),
                ],
                bond_orders=[WibergBondOrderRecord(
                    method="am1", values=[(0, 1, 1.2)])],
            )
        ],
    )

    expected_records = [argon, helium, chloride]
    store.store(expected_records)

    return store


class TestMoleculeStore:
    @pytest.fixture()
    def hcl_am1bcc(self):
        return MoleculeRecord(
            mapped_smiles="[Cl:2][H:1]",
            conformers=[
                ConformerRecord(
                    coordinates=np.arange(6).reshape((2, 3)),
                    partial_charges=[
                        PartialChargeRecord(
                            method="am1bcc", values=[0.25, 0.75])
                    ],
                    bond_orders=[
                        WibergBondOrderRecord(
                            method="am1", values=[(0, 1, 0.5)])
                    ],
                )
            ],
        )

    def test_db_version_property(self, tmp_path):
        """Tests that a version is correctly added to a new store."""
        store = MoleculeStore(f"{tmp_path}.sqlite")
        assert store.db_version == DB_VERSION

    def test_provenance_property(self, tmp_path):
        """Tests that a stores provenance can be set / retrieved."""

        store = MoleculeStore(f"{tmp_path}.sqlite")

        assert store.general_provenance == {}
        assert store.software_provenance == {}

        general_provenance = {"author": "Author 1"}
        software_provenance = {"psi4": "0.1.0"}
        store.set_provenance(general_provenance, software_provenance)
        assert store.general_provenance == general_provenance
        assert store.software_provenance == software_provenance

    def test_smiles_property(self, tmp_molecule_store):
        assert set(tmp_molecule_store.get_smiles()) == {
            "[Ar:1]",
            "[He:1]",
            "[Cl:1][Cl:2]",
        }

    def test_charge_methods_property(self, tmp_molecule_store):
        stored_methods = set(
            [x.value for x in tmp_molecule_store.get_charge_methods()])
        assert stored_methods == {"am1", "am1bcc"}

    def test_wbo_methods_property(self, tmp_molecule_store):
        assert set(tmp_molecule_store.get_wbo_methods()) == {
            WibergBondOrderMethod.AM1}

    def test_db_invalid_version(self, tmp_path):
        """Tests that the correct exception is raised when loading a store
        with an unsupported version."""

        store = MoleculeStore(f"{tmp_path}.sqlite")

        with store._get_session() as db:
            db_info = db.db.query(DBInformation).first()
            db_info.version = DB_VERSION - 1

        with pytest.raises(IncompatibleDBVersion) as error_info:
            MoleculeStore(f"{tmp_path}.sqlite")

        assert error_info.value.found_version == DB_VERSION - 1
        assert error_info.value.expected_version == DB_VERSION

    def test_store_partial_charge_data(self, tmp_path, hcl_am1bcc):

        store = MoleculeStore(f"{tmp_path}.sqlite")

        store.store(
            MoleculeRecord(
                mapped_smiles="[Cl:1][H:2]",
                conformers=[
                    ConformerRecord(
                        coordinates=np.arange(6).reshape((2, 3)),
                        partial_charges=[
                            PartialChargeRecord(
                                method="am1", values=[0.50, 1.50])
                        ],
                    )
                ],
            )
        )
        assert len(store) == 1
        assert store.get_charge_methods() == [ChargeMethod.AM1]

        store.store(hcl_am1bcc)

        assert len(store) == 1
        expected_charge_methods = {ChargeMethod.AM1, ChargeMethod.AM1BCC}
        assert {*store.get_charge_methods()} == expected_charge_methods

        record = store.retrieve()[0]
        assert len(record.conformers) == 1

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "am1bcc partial_charges already stored for [Cl:1][H:2]"),
        ):
            store.store(hcl_am1bcc)

        assert len(store) == 1
        assert {*store.get_charge_methods()} == expected_charge_methods

        record = store.retrieve()[0]
        assert len(record.conformers) == 1

    def test_store_bond_order_data(self, tmp_path, hcl_am1bcc):

        store = MoleculeStore(f"{tmp_path}.sqlite")

        store.store(hcl_am1bcc)
        assert store.get_smiles() == ["[Cl:2][H:1]"]
        assert len(store) == 1

        with pytest.raises(
            RuntimeError,
            match=re.escape("am1 bond_orders already stored for [Cl:2][H:1]"),
        ):
            store.store(
                MoleculeRecord(
                    mapped_smiles="[Cl:1][H:2]",
                    conformers=[
                        ConformerRecord(
                            coordinates=np.arange(6).reshape((2, 3)),
                            bond_orders=[
                                WibergBondOrderRecord(
                                    method="am1", values=[(0, 1, 0.5)]
                                )
                            ],
                        )
                    ],
                )
            )

        different_coordinates = MoleculeRecord(
            mapped_smiles="[Cl:2][H:1]",
            conformers=[
                ConformerRecord(
                    coordinates=np.zeros((2, 3)),
                    bond_orders=[
                        WibergBondOrderRecord(
                            method="am1", values=[(0, 1, 0.5)])
                    ],
                )
            ],
        )

        store.store(different_coordinates)

        assert len(store) == 1
        assert {*store.get_wbo_methods()} == {WibergBondOrderMethod.AM1}
        record = store.retrieve()[0]
        assert len(record.conformers) == 2

    @pytest.mark.parametrize(
        "partial_charge_method, bond_order_method, n_expected",
        [
            (None, None, 3),
            ("am1", None, 2),
            ("am1bcc", None, 2),
            ([], "am1", 1),
        ],
    )
    def test_retrieve_data(
        self, partial_charge_method, bond_order_method, n_expected, tmp_molecule_store
    ):

        retrieved_records = tmp_molecule_store.retrieve(
            partial_charge_methods=partial_charge_method,
            bond_order_methods=bond_order_method,
        )
        assert len(retrieved_records) == n_expected

        for record in retrieved_records:

            for conformer in record.conformers:

                assert partial_charge_method is None or all(
                    partial_charges.value == partial_charge_method
                    for partial_charges in conformer.partial_charges
                )

                assert bond_order_method is None or all(
                    bond_orders.value == bond_order_method
                    for bond_orders in conformer.bond_orders
                )

    def test_len(self, tmp_molecule_store):
        assert len(tmp_molecule_store) == 3
