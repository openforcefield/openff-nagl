
from typing import List, Tuple, Dict, Generator, Optional, TYPE_CHECKING, ClassVar, Any, Union
from rdkit import Chem
import json
import hashlib

import numpy as np

from gnn_charge_models.resonance.types import get_resonance_type, ResonanceAtomType, ResonanceTypeValue
from .paths import PathGenerator
from .bonds import increment_bond, decrement_bond

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule


class ResonanceEnumerator:

    _idx_property: ClassVar[str] = "original_idx"

    resonance_fragments: List[Dict[bytes, "FragmentEnumerator"]]
    acceptor_donor_fragments: List["FragmentEnumerator"]

    @classmethod
    def from_openff(cls, openff_molecule: "OFFMolecule"):
        rdmol = openff_molecule.to_rdkit()
        return cls(rdmol)

    def __init__(self, rdkit_molecule) -> None:
        self.original_rdkit_molecule = rdkit_molecule
        self.rdkit_molecule = Chem.Mol(rdkit_molecule)
        self.acceptor_donor_fragments = []
        self.resonance_fragments = []

        self.label_molecule()

    def label_molecule(self):
        for atom in self.rdkit_molecule.GetAtoms():
            atom.SetIntProp(self._idx_property, atom.GetIdx())

    @staticmethod
    def remove_hydrogens(rdmol: Chem.Mol):
        return Chem.RemoveAllHs(rdmol)

    @staticmethod
    def remove_uncharged_sp3_carbons(rdmol: Chem.Mol):
        query = Chem.MolFromSmarts("[#6+0X4]")
        indices = [
            ix[0]
            for ix in rdmol.GetSubstructMatches(query)
        ]

        editable = Chem.RWMol(rdmol)
        for ix in sorted(indices, reverse=True):
            editable.RemoveAtom(ix)

        return Chem.Mol(editable)

    def select_acceptor_donor_fragments(
        self,
        max_path_length: Optional[int] = None,
    ):
        rdmol = self.remove_hydrogens(self.rdkit_molecule)
        rdmol = self.remove_uncharged_sp3_carbons(rdmol)

        fragments = [
            FragmentEnumerator(rdfragment, max_path_length=max_path_length)
            for rdfragment in Chem.GetMolFrags(rdmol, asMols=True)
        ]

        acceptor_donor_fragments = []

        for fragment in fragments:
            if (
                fragment.acceptor_indices and
                fragment.donor_indices
            ):
                acceptor_donor_fragments.append(fragment)

        self.acceptor_donor_fragments = acceptor_donor_fragments
        return acceptor_donor_fragments

    def enumerate_resonance_fragments(
        self,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        # as_dicts: bool = False,
        include_all_transfer_pathways: bool = False,
    ) -> List[Dict[bytes, Union["FragmentEnumerator", Dict[str, Any]]]]:
        self.select_acceptor_donor_fragments(max_path_length=max_path_length)
        fragments: List[Dict[bytes, FragmentEnumerator]] = [
            fragment.enumerate_resonance_forms(
                lowest_energy_only=lowest_energy_only,
                include_all_transfer_pathways=include_all_transfer_pathways,
            )
            for fragment in self.acceptor_donor_fragments
        ]

        self.resonance_fragments = fragments
        # if as_dicts:
        #     fragments = [
        #         {
        #             k: v.to_resonance_dict()
        #             for k, v in fragment.items()
        #         }
        #         for fragment in fragments
        #     ]
        return fragments

    def get_resonance_atoms(self) -> Generator[List[Chem.rdchem.Atom], None, None]:
        for original_index in range(self.rdkit_molecule.n_atoms):
            rdatoms: List[Chem.rdchem.Atom] = [
                fragment.get_atom_with_original_idx(original_index)
                for fragment in self.resonance_fragments
                if original_index in fragment.original_resonance_types
            ]
            yield rdatoms


class FragmentEnumerator:
    def __init__(
        self,
        rdkit_molecule: Chem.Mol,
        max_path_length: Optional[int] = None,
    ):
        self.rdkit_molecule = Chem.Mol(rdkit_molecule)
        Chem.KekulizeMol(self.rdkit_molecule)
        self.current_to_original_atom_indices = {
            atom.GetIdx(): atom.GetIntProp(ResonanceEnumerator._idx_property)
            for atom in self.rdkit_molecule.GetAtoms()
        }
        self.original_to_current_atom_indices = {
            v: k for k, v in self.current_to_original_atom_indices.items()
        }

        self.path_generator = PathGenerator(rdkit_molecule)
        self.max_path_length = max_path_length
        self.acceptor_indices = []
        self.donor_indices = []

        self.resonance_types = self.get_resonance_types(rdkit_molecule)
        self.original_resonance_types = {
            self.current_to_original_atom_indices[k]: v
            for k, v in self.resonance_types.items()
        }
        for index, resonance_type in self.resonance_types.items():
            if resonance_type.type == ResonanceAtomType.Acceptor:
                self.acceptor_indices.append(index)
            elif resonance_type.type == ResonanceAtomType.Donor:
                self.donor_indices.append(index)

    def get_atom_with_original_idx(self, idx: int):
        current_idx = self.original_to_current_atom_indices[idx]
        return self.rdkit_molecule.GetAtomWithIdx(current_idx)

    def to_resonance_dict(self, include_bonds: bool = True) -> Dict[str, List[int]]:
        hash_dict = {
            "acceptor_indices": self.acceptor_indices,
            "donor_indices": self.donor_indices,
        }
        if include_bonds:
            hash_dict["bond_orders"] = [
                int(bond.GetBondTypeAsDouble())
                for bond in self.rdkit_molecule.GetBonds()
            ]
        return hash_dict

    def to_resonance_json(self, include_bonds: bool = True) -> str:
        return json.dumps(
            self.to_resonance_dict(include_bonds=include_bonds),
            sort_keys=True
        )

    def to_resonance_hash(self, include_bonds: bool = True) -> bytes:
        json_str = self.to_resonance_json(include_bonds=include_bonds)
        return hashlib.sha1(
            json_str.encode(),
            usedforsecurity=False
        ).digest()

    def get_integer_bond_order(self, source: int, target: int) -> int:
        bond = self.rdkit_molecule.GetBondBetweenAtoms(source, target)
        return int(bond.GetBondTypeAsDouble())

    def get_donor_acceptance_resonance_forms(
        self,
        donor: int,
        acceptor: int,
    ) -> Generator["FragmentEnumerator", None, None]:
        donor_to_acceptor = self.path_generator.all_odd_node_simple_paths(
            donor, acceptor, self.max_path_length
        )
        for path in donor_to_acceptor:
            if self.is_transfer_path(path):
                transferred = self.as_transferred(path)
                yield transferred

    def enumerate_donor_acceptor_resonance_forms(self) -> Generator["FragmentEnumerator", None, None]:
        for acceptor_index in self.acceptor_indices:
            for donor_index in self.donor_indices:
                form = self.get_donor_acceptance_resonance_forms(
                    donor_index, acceptor_index)
                yield form

    def enumerate_resonance_forms(
        self,
        include_all_transfer_pathways: bool = False,
        lowest_energy_only: bool = True
    ) -> Dict[bytes, "FragmentEnumerator"]:
        closed_forms: Dict[bytes, FragmentEnumerator] = {}
        open_forms: Dict[bytes, FragmentEnumerator] = {
            self.to_resonance_hash(include_bonds=True): self
        }

        while len(open_forms):

            current_forms: Dict[bytes, FragmentEnumerator] = {}

            for current_key, current_fragment in open_forms.items():
                if current_key in closed_forms:
                    continue
                closed_forms[current_key] = current_fragment

                for new_fragment in current_fragment.enumerate_donor_acceptor_resonance_forms():
                    new_key = new_fragment.to_resonance_hash(include_bonds=True)
                    if new_key not in closed_forms and new_key not in open_forms:
                        current_forms[new_key] = new_fragment

            open_forms = current_forms

        if not include_all_transfer_pathways:
            # Drop resonance forms that only differ due to bond order re-arrangements as we
            # aren't interested in e.g. ring resonance structures.
            closed_forms = {
                fragment.to_resonance_hash(include_bonds=False): fragment
                for fragment in closed_forms.values()
            }

        if lowest_energy_only:
            closed_forms = self.select_lowest_energy_forms(closed_forms)
        return closed_forms

    @staticmethod
    def select_lowest_energy_forms(forms: Dict[Any, "FragmentEnumerator"]) -> Dict[Any, "FragmentEnumerator"]:
        energies: Dict[Any, float] = {
            key: form.calculate_resonance_energy()
            for key, form in forms.items()
        }
        lowest = min(energies.values())
        lowest_forms = {
            key: forms[key]
            for key, energy in energies.items()
            if np.isclose(energy, lowest)
        }
        return lowest_forms

    def calculate_resonance_energy(self) -> float:
        return sum(res.energy for res in self.resonance_types.values())

    def transfer_electrons(self, path: List[int]) -> Chem.Mol:
        transferred = Chem.RWMol(self.rdkit_molecule)

        donor_index, acceptor_index = path[0], path[-1]
        for index in [donor_index, acceptor_index]:
            resonance_type = self.get_atom_resonance_type(index)
            conjugate_key = resonance_type.get_conjugate_key()
            rdatom = transferred.GetAtomWithIdx(index)

            rdatom.SetAtomicNum(conjugate_key.atomic_number)
            rdatom.SetFormalCharge(conjugate_key.formal_charge)

        edges = zip(path[:-1], path[1:])
        for bond_index, (i, j) in enumerate(edges):
            bond = transferred.GetBondBetweenAtoms(i, j)
            if bond_index % 2:
                decrement_bond(bond)
            else:
                increment_bond(bond)

        return Chem.Mol(transferred)

    def as_transferred(self, path: List[int], **kwargs) -> "FragmentEnumerator":
        transferred = self.transfer_electrons(path)
        return FragmentEnumerator(transferred, **kwargs)

    def is_transfer_path(self, path: List[Tuple[int, int]]) -> bool:
        edges = zip(path[:-1], path[1:])
        bond_orders = np.array([
            self.get_integer_bond_order(i, j)
            for i, j in edges
        ])

        deltas = bond_orders[1:] - bond_orders[:-1]
        return (
            np.all(deltas[::2] == 1) and
            np.all(deltas[1::2] == -1)
        )

    def get_resonance_types(self) -> Dict[int, ResonanceAtomType]:
        atom_types = {}
        for i in range(self.path_generator.n_atoms):
            try:
                atom_type = self.get_atom_resonance_type(i)
            except KeyError:
                pass
            else:
                atom_types[i] = atom_type
        return atom_types

    def get_atom_resonance_type(self, index: int) -> ResonanceTypeValue:
        atom = self.rdkit_molecule.GetAtomWithIdx(index)

        bonds = sorted(
            int(bond.GetBondTypeAsDouble())
            for bond in atom.GetBonds()
        )

        atom_type = get_resonance_type(
            element=atom.GetSymbol(),
            formal_charge=atom.GetFormalCharge(),
            bond_orders=tuple(bonds)
        )
        return atom_type
