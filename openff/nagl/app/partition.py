import copy
import random
from typing import Set, Tuple, TYPE_CHECKING, List, Iterable, Dict, Generator, Optional

import scipy.sparse
import numpy as np
import tqdm

from openff.utilities import requires_package
from openff.nagl.base.base import MutableModel 


if TYPE_CHECKING:
    from openff.nagl.storage.record import MoleculeRecord



class DatasetPartitioner(MutableModel):
    environments_by_element: Dict[str, Dict[str, Set[str]]]
    molecule_atom_fps: Dict[str, Set[str]]
    _all_environments: Optional[List[str]] = None
    _all_environment_indices: Optional[Dict[str, int]] = None
    _all_molecule_smiles: Optional[Dict[str, int]] = None
    _unrepresented_molecule_fp_matrix: Optional[scipy.sparse.lil_matrix] = None
    _atom_fp_count: Optional[Dict[str, int]] = None

    def _prepare_empty_vars(self):
        self._all_environments = None
        self._all_environment_indices = None
        self._all_molecule_smiles = None
        self._unrepresented_molecule_fp_matrix = None
        self._atom_fp_count = None

    @property
    def atom_fp_count(self):
        if self._atom_fp_count == None:
            self._atom_fp_count = {
                k: len(v)
                for env_dicts in self.environments_by_element.values()
                for k, v in env_dicts.items()
            }
        return self._atom_fp_count


    @property
    def all_environments(self):
        if self._all_environments is None:
            self._setup_molecule_fp_matrix()
        return self._all_environments
    
    @property
    def all_environment_indices(self):
        if self._all_environment_indices is None:
            self._setup_molecule_fp_matrix()
        return self._all_environment_indices

    @property
    def all_molecule_smiles(self):
        if self._all_molecule_smiles is None:
            self._setup_molecule_fp_matrix()
        return self._all_molecule_smiles

    @property
    def unrepresented_molecule_fp_matrix(self):
        if self._unrepresented_molecule_fp_matrix is None:
            self._setup_molecule_fp_matrix()
        return self._unrepresented_molecule_fp_matrix

    def __iadd__(self, other):
        if not isinstance(other, type(self)):
            raise ValueError("Can only add two DatasetPartitioners together.")
        for element, env_dicts in other.environments_by_element.items():
            if element not in self.environments_by_element:
                self.environments_by_element[element] = copy.deepcopy(env_dicts)
            else:
                for env, smiles in env_dicts.items():
                    if env not in self.environments_by_element[element]:
                        self.environments_by_element[element][env] = set()
                    self.environments_by_element[element][env] |= smiles
        self.molecule_atom_fps.update(other.molecule_atom_fps)

        self._prepare_empty_vars()
        return self

    
    @staticmethod
    @requires_package("rdkit")
    def get_atom_fingerprints(mapped_smiles: str) -> Set[Tuple[str, str]]:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from openff.toolkit.topology import Molecule
        from openff.nagl.utils.openff import capture_toolkit_warnings

        with capture_toolkit_warnings():
            environments = set()
            rdmol = Chem.AddHs(Chem.MolFromSmiles(mapped_smiles))
            # offmol = Molecule.from_smiles(
            #     mapped_smiles,
            #     allow_undefined_stereo=True
            # )
            for rdatom in rdmol.GetAtoms():
                fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(
                    rdmol,
                    maxLength=2,
                    nBits=2048,
                    nBitsPerEntry=4,
                    fromAtoms=[rdatom.GetIdx()]
                )
                z = rdatom.GetSymbol()
                environments.add((z, fp.ToBase64()))
            return environments
        
    @classmethod
    def from_molecule_records(cls, records: List["MoleculeRecord"]):
        smiles = {record.mapped_smiles for record in records}
        return cls.from_smiles(smiles)

    @classmethod
    def from_smiles(cls, all_smiles: List[str]):
        import tqdm
        from collections import defaultdict

        environments = defaultdict(lambda: defaultdict(set))
        molecule_fingerprints = {}
        all_smiles = sorted(all_smiles, key=len, reverse=True)
        for smiles in tqdm.tqdm(all_smiles, desc="Computing atom fingerprints in molecules"):
            envs = cls.get_atom_fingerprints(smiles)
            for symbol, fp in envs:
                environments[symbol][fp].add(smiles)
            molecule_fingerprints[smiles] = {x for _, x in envs}

        environments = {k: dict(v) for k, v in environments.items()}
        return cls(
            environments_by_element=environments,
            molecule_atom_fps=molecule_fingerprints
        )

        
    def _setup_molecule_fp_matrix(self):
        import numpy as np
        import scipy.sparse

        self._all_environments = [
            fp
            for fps in self.environments_by_element.values()
            for fp in fps
        ]

        self._all_environment_indices = {
            x: i
            for i, x in enumerate(self.all_environments)
        }
        
        indices = []
        
        self._all_molecule_smiles = {}
        # for i, (smiles, atom_fps) in tqdm.tqdm(
        #     enumerate(self.molecule_atom_fps.items()),
        #     desc="collecting matrix indices"
        # ):
        #     self.all_molecule_smiles[smiles] = i
        #     for fp in atom_fps:
        #         j = self.all_environment_indices[fp]
        #         indices.append((i, j))
        
        # I, J = tuple(zip(*indices))
        # data = np.ones_like(I)
        # matrix = scipy.sparse.coo_matrix((data, (I, J)), dtype=bool)
        n_molecules = len(self.molecule_atom_fps)
        n_fps = len(self.all_environments)
        matrix = scipy.sparse.coo_matrix(
            (n_molecules, n_fps),
            dtype=bool
        ).tolil()
        for i, (smiles, atom_fps) in tqdm.tqdm(
            enumerate(self.molecule_atom_fps.items()),
            desc="collecting matrix indices"
        ):
            self.all_molecule_smiles[smiles] = i
            for fp in atom_fps:
                j = self.all_environment_indices[fp]
                matrix[i, j] = 1
        self._unrepresented_molecule_fp_matrix = matrix #.tolil()
        
                
    def select_most_unrepresented_index(self, all_smiles: Iterable[str]) -> int:
        indices = [self.all_molecule_smiles[x] for x in all_smiles]
        sliced = self.unrepresented_molecule_fp_matrix[indices]
        return sliced.sum(axis=1).argmax()
    
    def remove_atom_fps_from_matrix(self, atom_fps: Iterable[str]):
        indices = [
            self.all_environment_indices[x]
            for x in atom_fps
        ]
        self.unrepresented_molecule_fp_matrix[:, indices] = 0
        
    def select_from_smiles(
        self,
        all_smiles: Iterable[str],
        n_environment_molecules: int = 4,
    ) -> List[str]:
        all_smiles = list(all_smiles)
        
        if len(all_smiles) <= n_environment_molecules:
            fps = set()
            for smiles in all_smiles:
                fps |= self.molecule_atom_fps[smiles]
            self.remove_atom_fps_from_matrix(fps)
            return all_smiles

        selected_smiles = []        
        for _ in range(n_environment_molecules):
            i = self.select_most_unrepresented_index(all_smiles)
            smiles = all_smiles[i]
            all_smiles.pop(i)
            atom_fps = self.molecule_atom_fps[smiles]
            self.remove_atom_fps_from_matrix(atom_fps)
            selected_smiles.append(smiles)
        return selected_smiles
    
    def count_atom_environments(self, all_smiles: Iterable[str]) -> Dict[str, int]:
        from collections import Counter
        
        counts = Counter([
            fp
            for smiles in all_smiles
            for fp in self.molecule_atom_fps[smiles]
        ])
        # for smiles in all_smiles:
        #     for fp in self.molecule_atom_fps[smiles]:
        #         counts[fp] += 1
        return counts
    
    def select_molecules(
        self,
        n_environment_molecules: int = 4,
        element_order: List[str] = ["S", "F", "Cl", "Br", "I", "P", "O", "N"]
    ) -> Set[str]:
        import tqdm

        selected_smiles = set()
        selected_fingerprints = set()
                
        # select all molecules with rare atom environments
        for el, envs in tqdm.tqdm(self.environments_by_element.items(), desc="processing rare environments"):
            for fp, all_smiles in envs.items():
                if len(all_smiles) <= n_environment_molecules:
                    selected_smiles |= all_smiles
                    selected_fingerprints.add(fp)
        
        # update fingerprint matrix
        self.remove_atom_fps_from_matrix(selected_fingerprints)
        
        # update environments_by_element
        environments = {
            el: {
                k: v
                for k, v in fps.items()
                if k not in selected_fingerprints
            }
            for el, fps in self.environments_by_element.items()
        }

        # for each of the specified atom environments (by element),
        # greedily select the molecules with the most unrepresented environments (all elements)
        for element in tqdm.tqdm(element_order, desc="processing ordered elements"):
            envs = environments.get(element, {})
            for fp, all_smiles in tqdm.tqdm(envs.items(), desc=f"processing {element} environments"):
                selected = self.select_from_smiles(
                    all_smiles,
                    n_environment_molecules=n_environment_molecules
                )
                selected_smiles |= set(selected)
    
        
        # for each of the remaining atom environments (by element)
        # if atom environment is not already adequately represented
        # greedily select molecules
        for element, envs in tqdm.tqdm(environments.items(), desc="processing remaining environments"):
            if element in element_order:
                continue
            seen_fps = self.count_atom_environments(selected_smiles)
            for fp, all_smiles in tqdm.tqdm(envs.items(), desc=f"processing {element} environments"):
                if seen_fps[fp] >= n_environment_molecules:
                    continue
                selected = self.select_from_smiles(
                    all_smiles,
                    n_environment_molecules=n_environment_molecules
                )
                selected_smiles |= set(selected)
                
        return selected_smiles

    @staticmethod
    def _get_counts_from_fractions(
        n: int,
        training_fraction: float,
        validation_fraction: float,
        test_fraction: float,
    ):
        n_test = int(np.round(n * test_fraction))
        n_validation = int(np.round(n * validation_fraction))
        n_training = int(np.round(n * training_fraction))

        while sum([n_validation, n_test, n_training]) > n:
            if n_test:
                n_test -= 1
            elif n_validation:
                n_validation -= 1
            else:
                n_training -= 1

        while sum([n_validation, n_test, n_training]) < n:
            if not n_training:
                n_training += 1
            elif not n_validation:
                n_validation += 1
            elif not n_test:
                n_test += 1
            else:
                n_training += 1
        return n_training, n_validation, n_test

    def _get_flattened_smiles_environments(
        self,
        element_order: List[str] = ["S", "F", "Cl", "Br", "I", "P", "O", "N"],
    ):

        def sort_smiles(smiles_dict):
            return [set().union(*map(set, smiles_dict.values()))]
            # return sorted(
            #     map(set, smiles_dict.values()),
            #     key=lambda x: sum(self._score_molecule_environment_richness(y) for y in x),
            #     reverse=True,
            # )

        flat_environments = []

        for el in element_order:
            if self.environments_by_element.get(el):
                smiles_sets = sort_smiles(self.environments_by_element[el])
                flat_environments.extend(smiles_sets)

        for el, env_dicts in self.environments_by_element.items():
            if el in element_order:
                continue
            smiles_sets = sort_smiles(env_dicts)
            flat_environments.extend(smiles_sets)
        
        return flat_environments

    def partition(
        self,
        training_fraction: float = 0.8,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        element_order: List[str] = ["S", "F", "Cl", "Br", "I", "P", "O", "N"],
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        import tqdm

        # normalize fractions
        total = training_fraction + validation_fraction + test_fraction
        training_fraction /= total
        validation_fraction /= total
        test_fraction /= total

        training = set()
        validation = set()
        test = set()
        all_datasets = [training, validation, test]

        seen_smiles = set()
        flat_environments = self._get_flattened_smiles_environments(element_order=element_order)

        for smiles in tqdm.tqdm(flat_environments, desc="partitioning"):
            # remove smiles from future sets
            smiles -= seen_smiles
            seen_smiles |= smiles

            # partition
            n = len(smiles)
            if not n:
                continue
            smiles_list = sorted(smiles, key=self._score_molecule_environment_richness, reverse=True)
            counts = self._get_counts_from_fractions(n, training_fraction, validation_fraction, test_fraction)
            for dataset, n in zip(all_datasets, counts):
                selected = set(self.select_from_smiles(smiles_list, n))
                dataset |= selected
                smiles_list = [s for s in smiles_list if s not in selected]
        
        return training, validation, test

    def _score_molecule_environment_richness(self, smiles):
        fps = self.molecule_atom_fps[smiles]
        return sum(1/self.atom_fp_count[fp] for fp in fps)



def stream_smiles_from_file(path: str) -> Generator[str, None, None]:
    from openff.nagl.utils.openff import stream_molecules_from_file
    from openff.nagl.storage.store import MoleculeStore

    if path.endswith("sqlite"):
        store = MoleculeStore(path)
        for record in store.retrieve():
            yield record.mapped_smiles
    else:
        for smiles in stream_molecules_from_file(path, as_smiles=True):
            yield smiles

    


