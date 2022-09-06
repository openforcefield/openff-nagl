from typing import Tuple, List

import dgl.function

from gnn_charge_models.base import ImmutableModel
from .molecule import DGLMolecule, DGLBase


class DGLMoleculeBatch(DGLBase):
    n_representations: Tuple[int, ...]
    n_atoms: Tuple[int, ...]

    @classmethod
    def from_dgl_molecules(cls, molecules: List[DGLMolecule]):
        graph = dgl.batch([molecule.graph for molecule in molecules])
        n_representations = tuple(
            molecule.n_representations
            for molecule in molecules
        )
        n_atoms = tuple(molecule.n_atoms for molecule in molecules)
        return cls(graph=graph, n_representations=n_representations, n_atoms=n_atoms)
