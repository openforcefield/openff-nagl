import copy
import functools
import random
import warnings
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import scipy.sparse
import tqdm
from openff.utilities import requires_package

from openff.nagl.base.base import MutableModel

if TYPE_CHECKING:
    from openff.nagl.storage.record import MoleculeRecord


class MoleculeSmiles(NamedTuple):
    smiles: str
    label: Any = None

class Fingerprint(NamedTuple):
    element: str
    fingerprint: str


class MoleculeAtomFingerprints:
    def to_tuple(self):
        return (self.smiles, self.atom_pair_fingerprints)

    def __hash__(self):
        return hash(self.to_tuple())
    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __init__(self, smiles, exclude_elements=("H",)):
        self.smiles = smiles
        fingerprints = tuple(sorted(self.generate_fingerprints(smiles, exclude_elements=exclude_elements)))
        self.atom_pair_fingerprints = fingerprints
        element_n_fingerprints = defaultdict(lambda: 0)
        for fp in self.atom_pair_fingerprints:
            element_n_fingerprints[fp.element] += 1
        self.element_n_fingerprints = dict(element_n_fingerprints)

    def get_n_fingerprints(self, exclude_elements=tuple()):
        counts = [
            v
            for k, v in self.element_n_fingerprints.items()
            if k not in exclude_elements
        ]
        return sum(counts)

    @staticmethod
    def generate_fingerprints(smiles: str, exclude_elements=("H",)):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        fingerprints = set()

        # rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            raise ValueError(f"SMILES could not be parsed: {smiles}")
        for rdatom in rdmol.GetAtoms():
            z = rdatom.GetSymbol()
            if z in exclude_elements:
                continue
            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(
                rdmol,
                maxLength=2,
                nBits=2048,
                nBitsPerEntry=4,
                fromAtoms=[rdatom.GetIdx()],
            )
            fingerprints.add(Fingerprint(z, fp.ToBase64()))

        return fingerprints


class FingerprintCollection:
    @classmethod
    def from_smiles(cls, smiles: List[str], exclude_elements: Tuple[str, ...] = ("H", )):
        fingerprints = []
        for smi in smiles:
            try:
                fp = MoleculeAtomFingerprints(smi, exclude_elements=exclude_elements)
            except ValueError:
                warnings.warn(f"Invalid SMILES {smi}")
            else:
                fingerprints.append(fp)

        return cls(fingerprints)

    def __init__(self, fingerprints):
        self.fingerprints = fingerprints

    @staticmethod
    def create_matrix(molecule_fingerprints, smiles_to_indices, fp_indices):
        n_molecules = len(smiles_to_indices)
        n_fingerprints = len(fp_indices)
        matrix = scipy.sparse.coo_matrix(
            (n_molecules, n_fingerprints), dtype=bool
        ).tolil()

        for fp in tqdm.tqdm(molecule_fingerprints, desc="creating matrix"):
            try:
                i_mol = smiles_to_indices[fp.smiles]
            except KeyError:
                continue
            i_fp = [fp_indices[x] for x in fp.atom_pair_fingerprints if x in fp_indices]
            matrix[i_mol, i_fp] = True
        return matrix

    def _select_atom_environments(
        self,
        molecule_fingerprints,
        atom_fingerprints,
        description: str,
        n_min_molecules: int,
        exclude_elements: Tuple[str, ...] = tuple(),
    ):
        smiles_to_index = {}
        index_to_smiles = {}
        n_other_fps = []
        for i, molfp in enumerate(molecule_fingerprints):
            index_to_smiles[i] = molfp.smiles
            smiles_to_index[molfp.smiles] = i
            n_fp = molfp.get_n_fingerprints(exclude_elements=exclude_elements)
            n_other_fps.append(n_fp)
        n_other_fps = np.array(n_other_fps)

        fp_indices = {x: i for i, x in enumerate(sorted(atom_fingerprints))}
        pool_matrix = self.create_matrix(
            molecule_fingerprints, smiles_to_index, fp_indices
        )

        selected_mol_indices = []
        for fp_i in tqdm.tqdm(fp_indices.values(), desc=description):
            pool = pool_matrix[:, fp_i]
            mol_ix, _ = pool.nonzero()
            if not sum(mol_ix.shape):
                continue
            n_unrepr_fps = np.array(pool_matrix[mol_ix].sum(axis=1)).flatten()
            n_unrepr_fps += n_other_fps[mol_ix]
            max_unrepr_ix = np.argsort(n_unrepr_fps)[::-1]
            max_unrepr = mol_ix[max_unrepr_ix[:n_min_molecules]]

            selected_mol_indices.extend(max_unrepr)
            pool_matrix[max_unrepr] = False
            pool_matrix[:, fp_i] = False

        return [index_to_smiles[i] for i in selected_mol_indices]

    def select_atom_environments(
        self,
        n_min_molecules: int = 4,
        element_order: List[str] = ["S", "F", "Cl", "Br", "I", "P", "O", "N", "C"],
    ) -> List[str]:

        # all_smiles = set()
        selected_smiles = set()
        all_fingerprints = {}
        fingerprint_smiles = defaultdict(set)
        element_fingerprints = defaultdict(set)

        for molfp in self.fingerprints:
            # all_smiles.add(molfp.smiles)
            all_fingerprints[molfp.smiles] = molfp
            for atfp in molfp.atom_pair_fingerprints:
                fingerprint_smiles[atfp].add(molfp.smiles)
                element_fingerprints[atfp.element].add(atfp)
        # first collect rare environments to minimize matrix sizes
        seen_fp = set()
        for fp, molsmiles in fingerprint_smiles.items():
            if len(molsmiles) <= n_min_molecules:
                selected_smiles |= molsmiles
                seen_fp.add(fp)
                element_fingerprints[fp.element] -= {fp}

        # print(f"Found {len(selected_smiles)} min smiles")
        # print(f"Found {len(seen_fp)} min fingerprints")

        fingerprint_smiles = {
            k: list(v) for k, v in fingerprint_smiles.items() if k not in seen_fp
        }

        excluded_elements = []
        for el in element_order:
            if el not in element_fingerprints:
                continue

            atom_fingerprints = element_fingerprints[el]
            molecule_smi: Set[str] = {
                smi for atfp in atom_fingerprints for smi in fingerprint_smiles[atfp]
            }
            molecule_fingerprints = [
                all_fingerprints[smi] for smi in sorted(molecule_smi - selected_smiles)
            ]

            el_smiles = self._select_atom_environments(
                molecule_fingerprints,
                atom_fingerprints,
                description=f"Searching element {el}",
                n_min_molecules=n_min_molecules,
                exclude_elements=excluded_elements,
            )
            selected_smiles |= set(el_smiles)
            excluded_elements.append(el)

        return sorted(selected_smiles)


class DatasetPartitioner:
    def __init__(self, smiles: Union[Iterable[str], Dict[str, Any]]):
        self.labelled_smiles: Dict[str, Any] = (
            smiles if isinstance(smiles, dict) else dict.fromkeys(smiles, None)
        )

        n_smiles = len(smiles)
        n_labelled = len(self.labelled_smiles)
        if not n_smiles == n_labelled:
            warnings.warn(
                "Found duplicate SMILES. "
                f"Partitioning {n_labelled} SMILES "
                f"from {n_smiles} original SMILES"
            )
        self.n_smiles = n_smiles

    def __iter__(self):
        for x in self.labelled_smiles:
            yield x

    def __sub__(self, other):
        if not isinstance(other, DatasetPartitioner):
            raise ValueError(
                "Maths is only supported between DatasetPartitioner objects"
            )

        labelled = {
            k: v
            for k, v in self.labelled_smiles.items()
            if k not in other.labelled_smiles
        }
        return type(self)(labelled)

    def __add__(self, other):
        if not isinstance(other, DatasetPartitioner):
            raise ValueError(
                "Maths is only supported between DatasetPartitioner objects"
            )

        labelled = dict(self.labelled_smiles)
        labelled.update(other.labelled_smiles)
        return type(self)(labelled)

    def select_atom_environments(
        self,
        n_min_molecules: int = 4,
        element_order: List[str] = ["S", "F", "Cl", "Br", "I", "P", "O", "N", "C"],
        exclude_elements: Tuple[str, ...] = ("H",)
    ) -> "DatasetPartitioner":
        exclude_elements = tuple([x for x in exclude_elements if x not in element_order])
        fp_collection = FingerprintCollection.from_smiles(self.labelled_smiles, exclude_elements=exclude_elements)
        selected = fp_collection.select_atom_environments(
            n_min_molecules=n_min_molecules, element_order=element_order
        )
        return self.smiles_with_labels(selected)

    def smiles_with_labels(self, smiles: Iterable[str]) -> "DatasetPartitioner":
        labelled = {x: self.labelled_smiles.get(x) for x in smiles}
        return type(self)(labelled)

    def select_diverse(
        self, n_molecules: int = 20000, seed: int = 42
    ) -> "DatasetPartitioner":
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
        from rdkit.DataStructs import DiceSimilarity
        from rdkit.SimDivFilters import MaxMinPicker

        if not n_molecules:
            return type(self)({})

        fingerprints = []
        smiles = list(self.labelled_smiles)
        for smi in tqdm.tqdm(smiles, desc="fingerprinting mols"):
            rdmol = Chem.MolFromSmiles(smi)
            if rdmol is None:
                warnings.warn(f"Invalid SMILES {smi}")
                continue
            fingerprints.append(GetMorganFingerprint(rdmol, 3))

        def dice_distance(i, j):
            return 1.0 - DiceSimilarity(fingerprints[i], fingerprints[j])

        picker = MaxMinPicker()
        selected_indices = picker.LazyPick(
            dice_distance, len(fingerprints), n_molecules, seed=int(seed)
        )
        selected_smiles = [smiles[i] for i in selected_indices]
        return self.smiles_with_labels(selected_smiles)

    def partition(
        self,
        training_fraction: float = 0.7,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.1,
        seed: int = 42,
    ) -> Tuple["DatasetPartitioner", "DatasetPartitioner", "DatasetPartitioner"]:
        # normalize fractions
        total = training_fraction + validation_fraction + test_fraction
        training_fraction /= total
        validation_fraction /= total
        test_fraction /= total

        # get counts
        n_test = int(np.round(self.n_smiles * test_fraction))
        n_validation = int(np.round(self.n_smiles * validation_fraction))
        n_training = self.n_smiles - n_test - n_validation

        pool = self
        test_set = pool.select_diverse(n_test, seed=seed)
        pool = pool - test_set
        validation_set = pool.select_diverse(n_validation, seed=seed)
        training_set = pool - validation_set

        return training_set, validation_set, test_set
