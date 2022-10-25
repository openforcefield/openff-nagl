from collections import defaultdict
import copy
import functools
import random
from typing import Set, Tuple, TYPE_CHECKING, List, Iterable, Dict, Generator, Optional, Union, Any, NamedTuple
import warnings

import scipy.sparse
import numpy as np
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

class MoleculeAtomFingerprints(NamedTuple):
    smiles: str
    atom_pair_fingerprints: Set[Fingerprint]

    @classmethod
    @requires_package("rdkit")
    def from_smiles(cls, smiles: str):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        fingerprints = set()
        elements = set()

        # rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            raise ValueError(f"SMILES could not be parsed: {smiles}")
        for rdatom in rdmol.GetAtoms():
            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(
                rdmol,
                maxLength=2,
                nBits=2048,
                nBitsPerEntry=4,
                fromAtoms=[rdatom.GetIdx()]
            )
            z = rdatom.GetSymbol()
            fingerprints.add(Fingerprint(z, fp.ToBase64()))

        return cls(smiles, fingerprints)


class FingerprintCollection:

    @classmethod
    def from_smiles(cls, smiles: List[str]):
        fingerprints = []
        for smi in smiles:
            try:
                fp = MoleculeAtomFingerprints.from_smiles(smi)
            except ValueError:
                warnings.warn(f"Invalid SMILES {smi}")
            else:
                fingerprints.append(fp)

        return cls(fingerprints)


    def __init__(self, fingerprints):
        self.fingerprints = fingerprints

        all_elements = defaultdict(set)
        self.fingerprint_elements = {}
        all_smiles = set()

        for molfp in fingerprints:
            all_smiles.add(molfp.smiles)
            for atfp in molfp.atom_pair_fingerprints:
                self.fingerprint_elements[atfp.fingerprint] = atfp.element
        
        self.all_smiles = sorted(all_smiles)
        self.index_to_smiles = dict(enumerate(self.all_smiles))
        self.smiles_indices = {x: i for i, x in self.index_to_smiles.items()}
        self.n_molecules = len(self.all_smiles)

        self.all_atom_fingerprints = sorted(fingerprint_elements)
        self.fp_indices = {x: i for i, x in enumerate(self.all_atom_fingerprints)}
        self.n_fingerprints = len(self.all_atom_fingerprints)

        element_indices = defaultdict(list)
        for fp, i in self.fp_indices.items():
            element_indices[self.fingerprint_elements[fp]].append(i)
        self.element_indices = dict(element_indices)


    def to_matrix(self):
        matrix = scipy.sparse.coo_matrix(
            (self.n_molecules, self.n_fingerprints),
            dtype=bool
        ).tolil()

        for fp in tqdm.tqdm(self.fingerprints, desc="creating matrix"):
            i_mol = self.smiles_indices[fp.smiles]
            i_fp = [self.fp_indices[x] for x in fp.atom_pair_fingerprints]
            matrix[i_mol, i_fp] = True
        return matrix

    @staticmethod
    def _select_most_unrepresented_molecules(pool_matrix, fp_i: int, n_min_molecules: int):
        pool = pool_matrix[:, fp_i]
        mol_indices, _ = pool.nonzero()
        n_unrepr_fps = np.array(pool_matrix[mol_indices].sum(axis=1))
        max_unrepr_ix = np.argsort(n_unrepr_fps.flatten())[::-1]
        max_unrepr = mol_indices[max_unrepr_ix[:n_min_molecules]]
        return max_unrepr


    def select_atom_environments(
        self,
        n_min_molecules: int = 4,
        element_order: List[str] = ["S", "F", "Cl", "Br", "I", "P", "O", "N", "C"]
    ) -> List[str]:
        pool_matrix = self.to_matrix()
        repr_matrix = scipy.sparse.coo_matrix(pool_matrix.shape).to_lil()
        selected_mol_indices = []

        # get all molecules with rare environments
        _, mask = (pool_matrix.sum(axis=0) <= n_min_molecules).nonzero()
        mol_indices, _ = pool_matrix[:, mask].nonzero()
        selected_mol_indices.extend(mol_indices)
        repr_matrix[mol_indices] = pool_matrix[mol_indices]
        pool_matrix[mol_indices] = False

        # go through elements in order
        unspecified = [el for el in self.element_indices if el not in element_order]
        for elements, desc in (
            [element_order, "searching ordered element"],
            [unspecified, "searching remaining element"],
        ):
            for el in elements:
                try:
                    fp_indices = self.element_indices[el]
                except KeyError:
                    continue

                for fp_i in tqdm.tqdm(
                    fp_indices,
                    desc=f"{desc} {el}"
                ):
                    if el not in element_order:
                        # skip environments already adequately represented
                        # for remaining elements
                        # (assumed to be mostly H)
                        n_mol = repr_matrix[:, fp_i].sum()
                        if n_mol >= n_min_molecules:
                            continue

                    # get molecules with highest number of as-yet unrepresented environments
                    pool = pool_matrix[:, fp_i]
                    mol_ix, _ = pool.nonzero()
                    n_unrepr_fps = np.array(pool_matrix[mol_ix].sum(axis=1))
                    max_unrepr_ix = np.argsort(n_unrepr_fps.flatten())[::-1]
                    max_unrepr = mol_ix[max_unrepr_ix[:n_min_molecules]]

                    selected_mol_indices.extend(max_unrepr)
                    repr_matrix[max_unrepr] = pool_matrix[max_unrepr]
                    pool_matrix[max_unrepr] = False
                    pool_matrix[:, fp_i] = False

        smiles = [self.index_to_smiles[i] for i in selected_mol_indices]
        return sorted(smiles)
        



class DatasetPartitioner:

    def __init__(
        self,
        smiles: Union[Iterable[str], Dict[str, Any]]
    ):
        self.labelled_smiles: Dict[str, Any] = (
            smiles if isinstance(smiles, dict)
            else dict.fromkeys(smiles, None)
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

    def __sub__(self, other):
        if not isinstance(other, DatasetPartitioner):
            raise ValueError("Maths is only supported between DatasetPartitioner objects")
        
        labelled = {
            k: v
            for k, v in self.labelled_smiles.items()
            if k not in other.labelled_smiles
        }
        return type(self)(labelled)

    def __add__(self, other):
        if not isinstance(other, DatasetPartitioner):
            raise ValueError("Maths is only supported between DatasetPartitioner objects")
        
        labelled = dict(self.labelled_smiles)
        labelled.update(other.labelled_smiles)
        return type(self)(labelled)


    def select_atom_environments(
        self,
        n_min_molecules: int = 4,
        element_order: List[str] = ["S", "F", "Cl", "Br", "I", "P", "O", "N", "C"]
    ) -> "DatasetPartitioner":
        fp_collection = FingerprintCollection.from_smiles(self.labelled_smiles)
        selected = fp_collection.select_atom_environments(
            n_min_molecules=n_min_molecules,
            element_order=element_order
        )
        return self.smiles_with_labels(selected)


    def smiles_with_labels(self, smiles: Iterable[str]) -> "DatasetPartitioner":
        labelled = {
            x: self.labelled_smiles.get(x)
            for x in smiles
        }
        return type(self)(labelled)


    def select_diverse(
        self,
        n_molecules: int = 20000,
        seed: int = 42
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
            fingerprints.append(fpGetMorganFingerprint(rdmol, 3))

        def dice_distance(i, j):
            return 1.0 - DiceSimilarity(fingerprints[i], fingerprints[j])
        
        picker = MaxMinPicker()
        selected_indices = picker.LazyPick(
            dice_distance,
            len(fingerprints),
            n_molecules,
            seed=seed
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







