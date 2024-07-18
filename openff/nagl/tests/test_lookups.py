import types
import pytest

import torch
import numpy as np
from numpy.testing import assert_allclose

from openff.toolkit import Molecule
from openff.nagl.lookups import (
    AtomPropertiesLookupTable,
    AtomPropertiesLookupTableEntry,
)


@pytest.fixture(scope="function")
def CNO2_entry():
    return AtomPropertiesLookupTableEntry(
        inchi="InChI=1/CH3NO2/c1-2(3)4/h1H3",
        mapped_smiles="[H:5][C:1]([H:6])([H:7])[N+:2](=[O:3])[O-:4]",
        property_value=[-0.103, 0.234, -0.209, -0.209, 0.096, 0.096, 0.096],
        provenance={"description": "test"}
    )


@pytest.fixture(scope="function")
def SH2_entry():
    return AtomPropertiesLookupTableEntry(
        inchi="InChI=1/H2S/h1H2",
        mapped_smiles="[H:2][S:1][H:3]",
        property_value=[-0.441, 0.22, 0.22],
        provenance={"description": "test"}
    )


class TestAtomPropertiesLookupTable:

    @pytest.fixture()
    def lookup_table(self, CNO2_entry, SH2_entry):
        return AtomPropertiesLookupTable(
            property_name="test",
            properties={
                CNO2_entry.inchi: CNO2_entry,
                SH2_entry.inchi: SH2_entry,
            }
        )

    def test_validate_property_lookup_table_conversion_from_list(
        self,
        CNO2_entry,
        SH2_entry
    ):
        lookup_table = AtomPropertiesLookupTable(
            property_name="test",
            properties=[CNO2_entry, SH2_entry],
        )

        assert isinstance(lookup_table.properties, types.MappingProxyType)
        assert len(lookup_table) == 2
        
        sh2_entry = lookup_table["InChI=1/H2S/h1H2"]
        assert sh2_entry.mapped_smiles == "[H:2][S:1][H:3]"
        assert sh2_entry.property_value == (-0.441, 0.22, 0.22)

    def test_creation_with_wrong_key(
        self,
        CNO2_entry,
    ):
        lookup_table = AtomPropertiesLookupTable(
            property_name="test",
            properties={
                "wrong_key": CNO2_entry,
            }
        )

        assert isinstance(lookup_table.properties, types.MappingProxyType)
        assert "InChI=1/CH3NO2/c1-2(3)4/h1H3" in lookup_table
        assert "wrong_key" not in lookup_table
        assert len(lookup_table) == 1

        
    
    def test_lookup(self, lookup_table):
        molecule = Molecule.from_mapped_smiles("[H:1][S:2][H:3]")
        properties = lookup_table.lookup(molecule)
        
        assert properties.shape == (3,)
        assert isinstance(properties, torch.Tensor)
        assert_allclose(properties.numpy(), np.array([0.22, -0.441, 0.22]))


    def test_lookup_failure(self, lookup_table):
        molecule = Molecule.from_smiles("CC")
        with pytest.raises(KeyError):
            lookup_table.lookup(molecule, match="Could not find")
