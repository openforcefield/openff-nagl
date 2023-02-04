from openff.nagl.molecule._utils import _get_openff_molecule_information


def test_get_openff_molecule_information(openff_methane_charged):
    # from openff.nagl.tests.testing.torch import assert_equal
    from numpy.testing import assert_equal

    info = _get_openff_molecule_information(openff_methane_charged)
    assert sorted(info.keys()) == ["atomic_number", "formal_charge", "idx"]
    assert_equal(info["idx"].numpy(), [0, 1, 2, 3, 4])
    assert_equal(info["formal_charge"].numpy(), [0, 0, 0, 0, 0])
    assert_equal(info["atomic_number"].numpy(), [6, 1, 1, 1, 1])
