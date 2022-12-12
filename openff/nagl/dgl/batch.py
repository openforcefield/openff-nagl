from typing import List, Tuple

import dgl.function

from .molecule import DGLBase, DGLMolecule

__all__ = [
    "DGLMoleculeBatch",
]


class DGLMoleculeBatch(DGLBase):
    n_representations: Tuple[int, ...]
    n_atoms: Tuple[int, ...]

    @classmethod
    def from_dgl_molecules(cls, molecules: List[DGLMolecule]):
        graph = dgl.batch([molecule.graph for molecule in molecules])
        n_representations = tuple(molecule.n_representations for molecule in molecules)
        n_atoms = tuple(molecule.n_atoms for molecule in molecules)
        return cls(graph=graph, n_representations=n_representations, n_atoms=n_atoms)

    @property
    def n_atoms_per_molecule(self):
        return self.n_atoms

    @property
    def n_representations_per_molecule(self):
        return self.n_representations
