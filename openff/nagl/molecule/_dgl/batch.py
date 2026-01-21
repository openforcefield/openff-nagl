from typing import List, Tuple

import torch

from openff.utilities import requires_package
from .molecule import DGLBase, DGLMolecule
from openff.nagl.molecule._base import BatchMixin


class DGLMoleculeBatch(BatchMixin, DGLBase):
    n_representations: Tuple[int, ...]
    n_atoms: Tuple[int, ...]

    def to(self, device: str):
        graph = self.graph.to(device)
        return type(self)(
            graph, n_representations=self.n_representations, n_atoms=self.n_atoms
        )

    @classmethod
    @requires_package("dgl")
    def from_dgl_molecules(cls, molecules: List[DGLMolecule]):
        import dgl

        graph = dgl.batch([molecule.graph for molecule in molecules])
        n_representations = tuple(molecule.n_representations for molecule in molecules)
        n_atoms = tuple(molecule.n_atoms for molecule in molecules)
        obj = cls(graph=graph, n_representations=n_representations, n_atoms=n_atoms)

        # concatenate pooling representations
        pooling_representations = {}
        keys = molecules[0]._pooling_representations.keys()
        for k in keys:
            all_indices = [
                molecule._pooling_representations[k] for molecule in molecules
            ]
            pooling_representations[k] = torch.cat(all_indices, dim=1)
        obj._pooling_representations.update(pooling_representations)
        return obj

    @requires_package("dgl")
    def unbatch(self) -> List[DGLMolecule]:
        import dgl

        return [
            DGLMolecule(g, n_repr)
            for g, n_repr in zip(
                dgl.unbatch(self.graph), self.n_representations_per_molecule
            )
        ]