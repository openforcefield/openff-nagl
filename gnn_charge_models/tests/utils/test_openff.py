import pytest


from gnn_charge_models.utils.openff import (
    get_unitless_charge,
    get_openff_molecule_bond_indices,
    get_openff_molecule_formal_charges,
    get_openff_molecule_information,
    map_indexed_smiles,
    normalize_molecule,
    get_best_rmsd,
    is_conformer_identical,
    smiles_to_inchi_key,
)


def test_get_unitless_charge(openff_methane_charged):
    formal = openff_methane_charged.atoms[1].formal_charge
    formal_charge = get_unitless_charge(formal, dtype=int)
    assert formal_charge == 0
    assert isinstance(formal_charge, int)

    partial = openff_methane_charged.atoms[1].partial_charge
    partial_charge = get_unitless_charge(partial)
    assert partial_charge == 0.1
    assert isinstance(partial_charge, float)


def test_get_openff_molecule_bond_indices(openff_methane_charged):
    bond_indices = get_openff_molecule_bond_indices(openff_methane_charged)
    assert bond_indices == [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4)
    ]


def test_get_openff_molecule_formal_charges(openff_methane_charged):
    formal_charges = get_openff_molecule_formal_charges(openff_methane_charged)
    assert formal_charges == [0, 0, 0, 0, 0]


def test_get_openff_molecule_information(openff_methane_charged):
    # from gnn_charge_models.tests.testing.torch import assert_equal
    from numpy.testing import assert_equal

    info = get_openff_molecule_information(openff_methane_charged)
    assert sorted(info.keys()) == ["atomic_number", "formal_charge", "idx"]
    assert_equal(info["idx"].numpy(), [0, 1, 2, 3, 4])
    assert_equal(info["formal_charge"].numpy(), [0, 0, 0, 0, 0])
    assert_equal(info["atomic_number"].numpy(), [6, 1, 1, 1, 1])
