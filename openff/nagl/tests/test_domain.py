import pytest

from openff.nagl.domains import ChemicalDomain


class TestChemicalDomain:
    @pytest.mark.parametrize(
        "elements",
        [
            (1, 6, 8),
            (1, 6, 8, 9, 17, 35),
        ],
    )
    def test_check_allowed_elements(self, elements, openff_methyl_methanoate):
        domain = ChemicalDomain(allowed_elements=elements)
        assert domain.check_allowed_elements(molecule=openff_methyl_methanoate)

    @pytest.mark.parametrize(
        "elements",
        [
            (8,),
            (6, 8, 9, 17, 35),
        ],
    )
    def test_check_allowed_elements_fail_noerr(
        self, elements, openff_methyl_methanoate
    ):
        domain = ChemicalDomain(allowed_elements=elements)
        assert not domain.check_allowed_elements(molecule=openff_methyl_methanoate)

    def test_check_allowed_elements_fail_err(self, openff_methyl_methanoate):
        domain = ChemicalDomain(allowed_elements=(8,))
        allowed, err = domain.check_allowed_elements(
            molecule=openff_methyl_methanoate, return_error_message=True
        )
        assert not allowed
        assert err == "Molecule contains forbidden element 6"

    @pytest.mark.parametrize(
        "patterns",
        [
            ("[*:1]#[*:2]",),
            ("[*:1]#[*:2]", "[#1:1]=[*:2]"),
        ],
    )
    def test_check_forbidden_patterns(self, patterns, openff_methyl_methanoate):
        domain = ChemicalDomain(forbidden_patterns=patterns)
        assert domain.check_forbidden_patterns(molecule=openff_methyl_methanoate)

    @pytest.mark.parametrize(
        "patterns",
        [
            ("[*:1]~[*:2]",),
            ("[#1:1]-[#6:2]", "[#1:1]#[*:2]"),
        ],
    )
    def test_check_forbidden_patterns_fail_noerr(
        self, patterns, openff_methyl_methanoate
    ):
        domain = ChemicalDomain(forbidden_patterns=patterns)
        assert not domain.check_forbidden_patterns(molecule=openff_methyl_methanoate)

    def test_check_forbidden_patterns_fail_err(self, openff_methyl_methanoate):
        domain = ChemicalDomain(forbidden_patterns=("[*:1]~[*:2]",))
        allowed, err = domain.check_forbidden_patterns(
            molecule=openff_methyl_methanoate, return_error_message=True
        )
        assert not allowed
        assert err == "Molecule contains forbidden SMARTS pattern [*:1]~[*:2]"

    def test_check_molecule_err(self, openff_methyl_methanoate):
        domain = ChemicalDomain(
            allowed_elements=(8, 6, 1), forbidden_patterns=("[*:1]~[*:2]",)
        )
        allowed, err = domain.check_molecule(
            molecule=openff_methyl_methanoate, return_error_message=True
        )
        assert not allowed
        assert err == "Molecule contains forbidden SMARTS pattern [*:1]~[*:2]"
