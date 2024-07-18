import types
import typing

import torch

from openff.nagl._base.base import ImmutableModel
from openff.nagl.utils._utils import is_iterable, potential_dict_to_list

try:
    from pydantic.v1 import Field, validator
except ImportError:
    from pydantic import Field, validator

if typing.TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


class PropertyProvenance(ImmutableModel):
    """
    Class for storing the provenance of a property
    """
    description: str = Field(
        description="A description of the provenance"
    )
    versions: dict[str, str] = Field(
        default_factory=dict,
        description="The versions of the relevant software packages used to compute the property"
    )

class BasePropertiesLookupTableEntry(ImmutableModel):
    inchi: str = Field(
        description="The InChI of the molecule"
    )
    provenance: PropertyProvenance = Field(
        description="The provenance of the property value"
    )

class AtomPropertiesLookupTableEntry(BasePropertiesLookupTableEntry):
    """
    Class for storing property lookup table entries
    """
    property_type: typing.Literal["atom"] = Field(
        default="atom",
        description="The type of the property"
    )
    
    mapped_smiles: str = Field(
        description="The mapped SMILES of the molecule"
    )

    property_value: tuple[float, ...] = Field(
        description=(
            "The values of the property, ordered according to mapped SMILES"
        )
    )

    def __len__(self):
        return len(self.property_value)

class BaseLookupTable(ImmutableModel):
    """
    Class for storing property lookup tables
    """

    property_name: str = Field(
        description="The name of the property"
    )


class AtomPropertiesLookupTable(BaseLookupTable):
    """
    Class for storing property lookup tables for atom properties
    """

    property_type: typing.Literal["atom"] = Field(
        default="atom",
        description="The type of the property"
    )

    properties: types.MappingProxyType[str, AtomPropertiesLookupTableEntry] = Field(
        description="The property lookup table"
    )

    @validator("properties", pre=True)
    def _convert_property_lookup_table(cls, v):
        """
        Do two things:
        
            1. Account for an iterable being passed instead of a mapping
            2. Ignore the keys of the mapping and re-generate them from inchi
            
        """
        v = potential_dict_to_list(v)
        if not is_iterable(v):
            raise ValueError("The property lookup table must be an iterable")
            
        if not all(isinstance(entry, AtomPropertiesLookupTableEntry) for entry in v):
            raise ValueError("All entries must be AtomPropertiesLookupTableEntry instances")

        return types.MappingProxyType({
            entry.inchi: entry
            for entry in v
        })
    
    def __len__(self) -> int:
        return len(self.properties)
    
    def __getitem__(self, key: str) -> AtomPropertiesLookupTableEntry:
        return self.properties[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self.properties

    def lookup(self, molecule: "Molecule") -> torch.Tensor:
        """
        Look up the property value for a molecule

        Parameters
        ----------
        molecule : openff.toolkit.topology.Molecule
            The molecule to look up
        
        Returns
        -------
        torch.Tensor
            The property values, in the order of the molecule's atoms
        
        Raises
        ------
        KeyError
            If the property value cannot be found for this molecule
        """
        from openff.toolkit.topology import Molecule

        inchi_key = molecule.to_inchi(fixed_hydrogens=True)
        try:
            entry = self.properties[inchi_key]
        except KeyError:
            raise KeyError(f"Could not find property value for molecule with InChI {inchi_key}")
        
        assert len(entry) == molecule.n_atoms

        # remap to query order
        entry_molecule = Molecule.from_mapped_smiles(
            entry.mapped_smiles,
            allow_undefined_stereo=True
        )
        is_isomorphic, query_to_entry_mapping = Molecule.are_isomorphic(
            molecule,
            entry_molecule,
            return_atom_map=True,
            # skip stereochemistry because matching inchi should be enough
            # atom_stereochemistry_matching=False,
            # bond_stereochemistry_matching=False,
        )
        assert is_isomorphic

        # remap the property values to the query order
        property_values = [
            entry.property_value[query_to_entry_mapping[atom_index]]
            for atom_index in range(molecule.n_atoms)
        ]
        return torch.tensor(property_values, dtype=torch.float32)





LookupTableEntryType = typing.Union[AtomPropertiesLookupTableEntry]
LookupTableType = typing.Union[AtomPropertiesLookupTable]

LOOKUP_TABLE_CLASSES = {
    "atom": AtomPropertiesLookupTable,
}


def _as_lookup_table(lookup_table_kwargs: dict) -> LookupTableType:
    """
    Convert a dictionary to a lookup table
    """
    if not isinstance(lookup_table_kwargs, BaseLookupTable):
        lookup_table_type = lookup_table_kwargs["property_type"]
        lookup_table_class = LOOKUP_TABLE_CLASSES[lookup_table_type]
        return lookup_table_class(**lookup_table_kwargs)
    return lookup_table_kwargs
