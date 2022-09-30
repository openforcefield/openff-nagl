import hashlib
import itertools
import json
from typing import (
    Any,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
from openff.toolkit.topology import Molecule as OFFMolecule
from rdkit import Chem

from openff.nagl.resonance.types import (
    ResonanceAtomType,
    ResonanceTypeValue,
    get_resonance_type,
)

from .bonds import decrement_bond, increment_bond
from .paths import PathGenerator

CURRENT_TRANSFER_PATH = "current_transfer_path"


def _remove_radicals(rdmol: Chem.Mol):
    for atom in rdmol.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumRadicalElectrons(0)


class ResonanceEnumerator:
    """Enumerate resonance structures for a molecule.

    This class is for recursively attempting to find all resonance structures of a molecule
    according to a modified version of the algorithm proposed by Gilson et al [1].

    Enumeration proceeds by:

    1. The molecule is turned into an RDKit molecule
    2. All hydrogen's and uncharged sp3 carbons are removed from the graph as these
       will not be involved in electron transfer.
    3. Disjoint sub-graphs are detected and separated out.
    4. Sub-graphs that don't contain at least 1 donor and 1 acceptor are discarded
    5. For each disjoint subgraph:
        a) The original v-charge algorithm is applied to yield the resonance structures
           of that subgraph.

    This will lead to ``M_i`` resonance structures for each of the ``N`` sub-graphs.

    If run with ``enumerate_resonance_forms``, then the resonance states in each
    sub-graph are returned in the form of a :class:`FragmentEnumerator`.
    If run with ``enumerate_resonance_molecules``, then the resonance states are
    combinatorially combined to yield molecule ojects in the requested type
    (either an OpenFF Molecule or an RDKit Mol).


    Notes
    -----
        * This method will strip all stereochemistry and aromaticity information from
          the input molecule.
        * The method only attempts to enumerate resonance forms that occur when a
          pair of electrons can be transferred along a conjugated path from a donor to
          an acceptor. Other types of resonance, e.g. different Kekule structures, are
          not enumerated.


    Parameters
    ----------
    openff_molecule: openff.toolkit.topology.Molecule


    Attributes
    ----------
    openff_molecule: openff.toolkit.topology.Molecule
        The molecule to enumerate resonance structures for.
    rdkit_molecule: rdkit.Chem.Mol
        The molecule in RDKit format
    acceptor_donor_fragments: List[FragmentEnumerator]
        The fragments of the molecule that contain acceptor or donor atoms.
    resonance_fragments: Dict[bytes, FragmentEnumerator]
        The resonance fragments of the molecule.


    References
    ----------
    [1] Gilson, Michael K., Hillary SR Gilson, and Michael J. Potter. "Fast
        assignment of accurate partial atomic charges: an electronegativity
        equalization method that accounts for alternate resonance forms." Journal of
        chemical information and computer sciences 43.6 (2003): 1982-1997.
    """

    _idx_property: ClassVar[str] = "original_idx"

    resonance_fragments: List[Dict[bytes, "FragmentEnumerator"]]
    acceptor_donor_fragments: List["FragmentEnumerator"]

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        mapped: bool = False,
        allow_undefined_stereo: bool = True,
    ):
        func = OFFMolecule.from_smiles
        if mapped:
            func = OFFMolecule.from_mapped_smiles
        offmol = func(smiles, allow_undefined_stereo=allow_undefined_stereo)
        return cls(offmol)

    def __init__(self, openff_molecule: "OFFMolecule"):
        self.openff_molecule = openff_molecule
        self.rdkit_molecule = openff_molecule.to_rdkit()
        Chem.Kekulize(self.rdkit_molecule)
        self.acceptor_donor_fragments = []
        self.resonance_fragments = {}

        self.label_molecule()

    def label_molecule(self):
        for atom in self.rdkit_molecule.GetAtoms():
            atom.SetIntProp(self._idx_property, atom.GetIdx())

    @staticmethod
    def remove_hydrogens(rdmol: Chem.Mol):
        return Chem.RemoveAllHs(rdmol, sanitize=False)

    @staticmethod
    def remove_uncharged_sp3_carbons(rdmol: Chem.Mol):
        query = Chem.MolFromSmarts("[#6+0X4]")
        indices = [ix[0] for ix in rdmol.GetSubstructMatches(query)]
        indices_set = set(indices)

        editable = Chem.RWMol(rdmol)
        for atom in editable.GetAtoms():
            neighbors = {x.GetIdx() for x in atom.GetNeighbors()}
            overlap = neighbors & indices_set
            if overlap:
                atom.SetNumExplicitHs(len(overlap))
        for ix in sorted(indices, reverse=True):
            editable.RemoveAtom(ix)
        return editable
        # return Chem.Mol(editable)

    def _clean_molecule(self):
        rdmol = self.remove_hydrogens(self.rdkit_molecule)
        rdmol = self.remove_uncharged_sp3_carbons(rdmol)
        # _remove_radicals(rdmol)
        return rdmol

    def select_acceptor_donor_fragments(
        self,
        max_path_length: Optional[int] = None,
    ):
        """Select the fragments of the molecule that contain acceptor or donor atoms.

        Parameters
        ----------
        max_path_length: int, optional
            The maximum number of bonds between a donor and acceptor to
            consider.

        Returns
        -------
        List[FragmentEnumerator]
        """
        rdmol = self._clean_molecule()

        fragments = [
            FragmentEnumerator(rdfragment, max_path_length=max_path_length)
            for rdfragment in Chem.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
        ]

        acceptor_donor_fragments = []

        for fragment in fragments:
            if fragment.acceptor_indices and fragment.donor_indices:
                acceptor_donor_fragments.append(fragment)

        self.acceptor_donor_fragments = acceptor_donor_fragments
        return self.acceptor_donor_fragments

    def enumerate_resonance_fragments(
        self,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
    ) -> List[Dict[bytes, "FragmentEnumerator"]]:
        """Enumerate the resonance fragments of the molecule.

        Parameters
        ----------
        lowest_energy_only: bool, optional
            Whether to only return the resonance forms with the lowest "energy"
        max_path_length: int, optional
            The maximum number of bonds between a donor and acceptor to
            consider.
        include_all_transfer_pathways: bool, optional
            Whether to include resonance forms that have
            the same formal charges but have different arrangements of bond orders. Such
            cases occur when there exists multiple electron transfer pathways between
            electron donor-acceptor pairs e.g. in cyclic systems.

        Returns
        -------
        List[Dict[bytes, FragmentEnumerator]]
            Each item in the list is a dictionary mapping the resonance hash
            to the actual FragmentEnumerator.
        """
        self.select_acceptor_donor_fragments(max_path_length=max_path_length)
        fragments: List[Dict[bytes, FragmentEnumerator]] = [
            fragment.enumerate_resonance_forms(
                lowest_energy_only=lowest_energy_only,
                include_all_transfer_pathways=include_all_transfer_pathways,
            )
            for fragment in self.acceptor_donor_fragments
        ]

        self.resonance_fragments = fragments
        return fragments

    def as_fragment(self) -> "FragmentEnumerator":
        return FragmentEnumerator(self.rdkit_molecule)

    def enumerate_resonance_molecules(
        self,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
        moleculetype: Union[Type[Chem.rdchem.Mol],
                            Type[OFFMolecule]] = Chem.rdchem.Mol,
    ) -> Union[List[Chem.rdchem.Mol], List[OFFMolecule]]:
        """Combinatorially enumerate all resonance forms of the molecule.

        Parameters
        ----------
        lowest_energy_only: bool, optional
            Whether to only return the resonance forms with the lowest "energy"
        max_path_length: int, optional
            The maximum number of bonds between a donor and acceptor to
            consider.
        include_all_transfer_pathways: bool, optional
            Whether to include resonance forms that have
            the same formal charges but have different arrangements of bond orders. Such
            cases occur when there exists multiple electron transfer pathways between
            electron donor-acceptor pairs e.g. in cyclic systems.
        moleculetype: type, optional
            The type of molecule to return.

        Returns
        -------
        Union[List[rdkit.Chem.rdchem.Mol], List[openff.toolkit.topology.Molecule]]
        """

        fragments = self.enumerate_resonance_fragments(
            lowest_energy_only=lowest_energy_only,
            max_path_length=max_path_length,
            include_all_transfer_pathways=include_all_transfer_pathways,
        )
        return self.build_resonance_molecules(fragments, moleculetype=moleculetype)

    def build_resonance_molecules(
        self,
        fragments,
        moleculetype: Union[Type[Chem.rdchem.Mol],
                            Type[OFFMolecule]] = Chem.rdchem.Mol,
    ) -> Union[List[Chem.rdchem.Mol], List[OFFMolecule]]:
        rdkit_molecules = []
        for combination in itertools.product(*fragments):
            molecule = self.as_fragment()
            for fragment_index, unit_key in enumerate(combination):
                unit = fragments[fragment_index][unit_key]
                # we can skip the check since we just made these fragments
                molecule._substitute_fragment(unit)
            Chem.SanitizeMol(molecule.rdkit_molecule)
            rdkit_molecules.append(molecule.rdkit_molecule)

        if moleculetype is Chem.rdchem.Mol:
            return rdkit_molecules
        elif moleculetype is OFFMolecule:
            return [
                OFFMolecule.from_rdkit(rdmol, allow_undefined_stereo=True)
                for rdmol in rdkit_molecules
            ]
        raise NotImplementedError(f"Molecule type {moleculetype} not supported")

    def get_resonance_atoms(self) -> Generator[List[Chem.rdchem.Atom], None, None]:
        molecules = self.build_resonance_molecules(self.resonance_fragments)

        for original_index in range(self.rdkit_molecule.GetNumAtoms()):
            rdatoms: List[Chem.rdchem.Atom] = [
                rdmol.GetAtomWithIdx(original_index)
                for rdmol in molecules
            ]
            yield rdatoms

    @classmethod
    def _get_original_index(cls, rdkit_atom):
        try:
            return rdkit_atom.GetIntProp(cls._idx_property)
        except KeyError:
            return rdkit_atom.GetIdx()


class FragmentEnumerator:
    """A class for enumerating the resonance forms of a fragment of a molecule.

    If this has been returned as a result of ResonanceEnumerator,
    the fragment is the ``rdkit_molecule`` attribute.
    It can be mapped back to the original molecule using the
    ``current_to_original_atom_indices`` and
    ``original_to_current_atom_indices`` attributes.


    Parameters
    ----------
    rdkit_molecule: rdkit.Chem.rdchem.Mol
        The molecule to enumerate the resonance forms of.
    max_path_length: int, optional
        The maximum number of bonds between a donor and acceptor to
        consider.
    clean_molecule: bool, optional
        Whether to clean the valence of the molecule by removing radicals
        and kekulizing it.


    Attributes
    ----------
    rdkit_molecule: rdkit.Chem.rdchem.Mol
        The molecule to enumerate the resonance forms of.
    current_to_original_atom_indices: Dict[int, int]
        A mapping from the current atom indices of ``rdkit_molecule``
        to the original atom indices of the larger molecule in the
        :class:`ResonanceEnumerator`.
    original_to_current_atom_indices: Dict[int, int]
        A mapping from the original atom indices of the larger molecule
        in the :class:`ResonanceEnumerator` to the current atom indices
    """

    def __init__(
        self,
        rdkit_molecule: Chem.Mol,
        max_path_length: Optional[int] = None,
        clean_molecule: bool = False,
    ):
        self.rdkit_molecule = Chem.Mol(rdkit_molecule)
        if clean_molecule:
            _remove_radicals(self.rdkit_molecule)
            Chem.Kekulize(self.rdkit_molecule)

        self.current_to_original_atom_indices = {
            atom.GetIdx(): ResonanceEnumerator._get_original_index(atom)
            for atom in self.rdkit_molecule.GetAtoms()
        }
        self.original_to_current_atom_indices = {
            v: k for k, v in self.current_to_original_atom_indices.items()
        }

        self.path_generator = PathGenerator(self.rdkit_molecule)
        self.max_path_length = max_path_length
        self.acceptor_indices = []
        self.donor_indices = []

        self.resonance_types = self.get_resonance_types()
        self.original_resonance_types = {
            self.current_to_original_atom_indices[k]: v
            for k, v in self.resonance_types.items()
        }
        for index, resonance_type in self.resonance_types.items():
            if resonance_type.type == ResonanceAtomType.Acceptor.value:
                self.acceptor_indices.append(index)
            elif resonance_type.type == ResonanceAtomType.Donor.value:
                self.donor_indices.append(index)

    def substitute_fragment(self, fragment: "FragmentEnumerator"):
        original_indices = set(self.original_to_current_atom_indices)
        fragment_indices = set(fragment.original_to_current_atom_indices)
        if not fragment_indices.issubset(original_indices):
            raise ValueError(
                "Fragment original indices must be a subset "
                "of the original molecule indices to substitute in. "
                f"{fragment_indices} is not a subset of {original_indices}"
            )

        self._substitute_fragment(fragment)

    def _substitute_fragment(self, fragment: "FragmentEnumerator"):
        fragment_to_self = {
            fragment_index: self.original_to_current_atom_indices[original_index]
            for fragment_index, original_index in fragment.current_to_original_atom_indices.items()
        }

        for fragment_index, current_index in fragment_to_self.items():
            rdatom = self.rdkit_molecule.GetAtomWithIdx(current_index)
            fragment_atom = fragment.rdkit_molecule.GetAtomWithIdx(
                fragment_index)
            self._substitute_atom(rdatom, fragment_atom)

        for fragment_bond in fragment.rdkit_molecule.GetBonds():
            fragment_i = fragment_bond.GetBeginAtomIdx()
            fragment_j = fragment_bond.GetEndAtomIdx()
            current_i = fragment_to_self[fragment_i]
            current_j = fragment_to_self[fragment_j]

            self_bond = self.rdkit_molecule.GetBondBetweenAtoms(
                current_i, current_j)
            self._substitute_bond(self_bond, fragment_bond)

    @staticmethod
    def _substitute_atom(current, new):
        current.SetFormalCharge(new.GetFormalCharge())

    @staticmethod
    def _substitute_bond(current, new):
        current.SetBondType(new.GetBondType())

    def get_atom_with_original_idx(self, idx: int):
        current_idx = self.original_to_current_atom_indices[idx]
        return self.rdkit_molecule.GetAtomWithIdx(current_idx)

    def to_resonance_dict(
        self, include_bonds: bool = True, include_formal_charges: bool = False
    ) -> Dict[str, List[int]]:
        hash_dict = {
            "acceptor_indices": self.acceptor_indices,
            "donor_indices": self.donor_indices,
        }
        if include_bonds:
            hash_dict["bond_orders"] = self.get_bond_orders()

        if include_formal_charges:
            hash_dict["formal_charges"] = self.get_formal_charges()
        return hash_dict

    def to_resonance_json(
        self, include_bonds: bool = True, include_formal_charges: bool = False
    ) -> str:
        return json.dumps(
            self.to_resonance_dict(
                include_bonds=include_bonds,
                include_formal_charges=include_formal_charges,
            ),
            sort_keys=True,
        )

    def to_resonance_hash(
        self, include_bonds: bool = True, include_formal_charges: bool = False
    ) -> bytes:
        json_str = self.to_resonance_json(
            include_bonds=include_bonds, include_formal_charges=include_formal_charges
        )
        return hashlib.sha1(json_str.encode(), usedforsecurity=False).digest()

    def get_integer_bond_order(self, source: int, target: int) -> int:
        bond = self.rdkit_molecule.GetBondBetweenAtoms(source, target)
        return int(bond.GetBondTypeAsDouble())

    def get_donor_acceptance_resonance_forms(
        self,
        donor: int,
        acceptor: int,
    ) -> Generator["FragmentEnumerator", None, None]:
        for path in self.get_transfer_paths(donor, acceptor):
            transferred = self.as_transferred(path)
            yield transferred

    def get_transfer_paths(
        self, donor: int, acceptor: int
    ) -> Generator[Tuple[int, ...], None, None]:
        donor_to_acceptor = self.path_generator.all_odd_node_simple_paths(
            donor, acceptor, self.max_path_length
        )
        for path in donor_to_acceptor:
            if self.is_transfer_path(path):
                yield path

    def enumerate_donor_acceptor_resonance_forms(
        self,
    ) -> Generator["FragmentEnumerator", None, None]:
        for acceptor_index in self.acceptor_indices:
            for donor_index in self.donor_indices:
                for form in self.get_donor_acceptance_resonance_forms(
                    donor_index, acceptor_index
                ):
                    yield form

    def enumerate_resonance_forms(
        self,
        include_all_transfer_pathways: bool = False,
        lowest_energy_only: bool = True,
    ) -> Dict[bytes, "FragmentEnumerator"]:

        self_hash = self.to_resonance_hash(include_bonds=True)
        open_forms: Dict[bytes, FragmentEnumerator] = {self_hash: self}
        closed_forms: Dict[bytes, FragmentEnumerator] = {}

        while len(open_forms):
            current_forms: Dict[bytes, FragmentEnumerator] = {}

            for current_key, current_fragment in open_forms.items():
                if current_key in closed_forms:
                    continue
                closed_forms[current_key] = current_fragment

                for (
                    new_fragment
                ) in current_fragment.enumerate_donor_acceptor_resonance_forms():
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
    def select_lowest_energy_forms(
        forms: Dict[Any, "FragmentEnumerator"]
    ) -> Dict[Any, "FragmentEnumerator"]:
        energies: Dict[Any, float] = {
            key: form.calculate_resonance_energy() for key, form in forms.items()
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
        # _remove_radicals(transferred)
        # Chem.Kekulize(transferred)

        donor_index, acceptor_index = path[0], path[-1]
        for index in [donor_index, acceptor_index]:
            resonance_type = self.get_atom_resonance_type(index)
            conjugate_key = resonance_type.get_conjugate_key()
            rdatom = transferred.GetAtomWithIdx(index)

            rdatom.SetAtomicNum(conjugate_key.atomic_number)
            rdatom.SetFormalCharge(conjugate_key.formal_charge)

        edges = list(zip(path[:-1], path[1:]))
        for bond_index, (i, j) in enumerate(edges):
            bond = transferred.GetBondBetweenAtoms(i, j)
            if bond_index % 2:
                decrement_bond(bond)
            else:
                increment_bond(bond)
        for atom in transferred.GetAtoms():
            atom.UpdatePropertyCache()

        return Chem.Mol(transferred)

    def as_transferred(self, path: List[int], **kwargs) -> "FragmentEnumerator":
        transferred = self.transfer_electrons(path)
        return FragmentEnumerator(transferred, **kwargs)

    def is_transfer_path(self, path: List[Tuple[int, int]]) -> bool:
        edges = zip(path[:-1], path[1:])
        bond_orders = np.array(
            [self.get_integer_bond_order(i, j) for i, j in edges])

        deltas = bond_orders[1:] - bond_orders[:-1]
        return np.all(deltas[::2] == 1) and np.all(deltas[1::2] == -1)

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

        bonds = [int(bond.GetBondTypeAsDouble()) for bond in atom.GetBonds()]
        atom.UpdatePropertyCache()
        n_imp_h = atom.GetNumImplicitHs()
        bonds += [1] * n_imp_h

        orders = tuple(sorted(bonds))
        atom_type = get_resonance_type(
            element=atom.GetSymbol(),
            formal_charge=atom.GetFormalCharge(),
            bond_orders=orders,
        )

        return atom_type

    def get_formal_charges(self):
        return [atom.GetFormalCharge() for atom in self.rdkit_molecule.GetAtoms()]

    def get_bond_orders(self):
        return {
            i: int(bond.GetBondTypeAsDouble())
            for i, bond in enumerate(self.rdkit_molecule.GetBonds())
        }
        # bond_orders = {}
        # for bond in self.rdkit_molecule.GetBonds():
        #     i = bond.GetBeginAtomIdx()
        #     j = bond.GetEndAtomIdx()
        #     if with_original_index:
        #         i = self.current_to_original_atom_indices[i]
        #         j = self.current_to_original_atom_indices[j]
        #     key = (i, j) if i < j else (j, i)
        #     bond_orders[key] = int(bond.GetBondTypeAsDouble())
        # return bond_orders
