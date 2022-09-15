import pytest
import numpy as np
from pydantic import ValidationError

from openff.toolkit.topology.molecule import Molecule as OFFMolecule
from gnn_charge_models.storage.record import (
    ChargeMethod,
    PartialChargeRecord,
    WibergBondOrderMethod,
    WibergBondOrderRecord,
    WibergBondOrder,
    MoleculeRecord,
    ConformerRecord,
)


class TestPartialChargeRecord:
    def test_type_coercion(self):
        record = PartialChargeRecord(method="am1", values=[0.1])
        assert record.method is ChargeMethod.AM1
        assert isinstance(record.values, np.ndarray)


class TestWibergBondOrderRecord:
    def test_type_coercion(self):
        record = WibergBondOrderRecord(method="am1", values=[(0, 1, 0.1)])
        assert record.method is WibergBondOrderMethod.AM1
        assert isinstance(record.values, list)

        value = record.values[0]
        assert isinstance(value, WibergBondOrder)
        assert value.atom_index_1 == 0
        assert value.atom_index_2 == 1
        assert np.isclose(value.bond_order, 0.1)


class TestConformerRecord:
    def test_charges_and_bonds(self):
        record = ConformerRecord(
            coordinates=np.ones((4, 3)),
            partial_charges=[
                PartialChargeRecord(method="am1", values=[0.1, 0.2, 0.3, 0.4]),
                PartialChargeRecord(method="am1bcc", values=[
                                    1.0, 2.0, 3.0, 4.0]),
            ],
            bond_orders=[
                WibergBondOrderRecord(
                    method="am1", values=[(0, 1, 0.1)])
            ],
        )

        charges = {k: tuple(v.values)
                   for k, v in record.partial_charges.items()}
        assert charges == {
            ChargeMethod.AM1: (0.1, 0.2, 0.3, 0.4),
            ChargeMethod.AM1BCC: (1.0, 2.0, 3.0, 4.0),
        }

        bonds = {k: v.values for k, v in record.bond_orders.items()}
        assert bonds == {
            WibergBondOrderMethod.AM1: [(0, 1, 0.1), ],
        }

    @pytest.mark.parametrize("input_coordinates", [
        np.arange(6),
        np.arange(6).reshape((3, 2)),
    ])
    def test_validate_coordinates(self, input_coordinates):
        record = ConformerRecord(coordinates=input_coordinates)
        assert record.coordinates.shape == (2, 3)

    @pytest.mark.parametrize("input_coordinates", [
        np.arange(5),
        np.arange(4).reshape((-1, 2))
    ])
    def test_error_for_invalid_coordinates(self, input_coordinates):
        with pytest.raises(ValidationError, match="must be re-shapable"):
            ConformerRecord(coordinates=input_coordinates)


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

    def test_average_partial_charges(self):

        record = MoleculeRecord(
            mapped_smiles="[C:1]([H:2])([H:3])([H:4])",
            conformers=[
                ConformerRecord(
                    coordinates=np.ones((4, 3)),
                    partial_charges=[
                        PartialChargeRecord(method="am1", values=[
                                            0.1, 0.2, 0.3, 0.4]),
                    ],
                ),
                ConformerRecord(
                    coordinates=np.zeros((4, 3)),
                    partial_charges=[
                        PartialChargeRecord(method="am1", values=[
                                            0.3, 0.4, 0.5, 0.6]),
                    ],
                ),
            ],
        )

        average_charges = record.average_partial_charges("am1")
        assert isinstance(average_charges, np.ndarray)
        assert len(average_charges) == 4
        assert np.allclose(average_charges, (0.2, 0.3, 0.4, 0.5))

    def test_reorder(self):

        original_coordinates = np.arange(6).reshape((2, 3))

        original_record = MoleculeRecord(
            mapped_smiles="[Cl:2][H:1]",
            conformers=[
                ConformerRecord(
                    coordinates=original_coordinates,
                    partial_charges=[PartialChargeRecord(
                        method="am1", values=[0.5, 1.5])],
                    bond_orders=[
                        WibergBondOrderRecord(
                            method="am1", values=[(0, 1, 0.2)])
                    ],
                )
            ],
        )

        reordered_record = original_record.reorder("[Cl:1][H:2]")
        assert reordered_record.mapped_smiles == "[Cl:1][H:2]"

        reordered_conformer = reordered_record.conformers[0]

        assert np.allclose(
            reordered_conformer.coordinates,
            np.flipud(original_coordinates)
        )

        assert np.allclose(
            reordered_conformer.partial_charges[ChargeMethod.AM1].values,
            [1.5, 0.5]
        )
        assert np.allclose(
            reordered_conformer.bond_orders[WibergBondOrderMethod.AM1].values,
            [(1, 0, 0.2)]
        )
