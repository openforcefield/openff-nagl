
from typing import List

from openff.nagl.molecule._base import NAGLMoleculeBase, MoleculeMixin, BatchMixin

class NXMolecule(MoleculeMixin, NAGLMoleculeBase):
    def to_homogenous(self):
        return self.graph.to_homogenous()
    
    def to(self, device: str):
        return type(self)(self.graph)

    @property
    def n_graph_nodes(self):
        return int(self.graph.graph.number_of_nodes())

    @property
    def n_graph_edges(self):
        return int(self.graph.graph.number_of_edges())
    

class NXMoleculeBatch(BatchMixin, NAGLMoleculeBase):
    
    @classmethod
    def from_nx_molecules(cls, molecules: List[NXMolecule]):
        from openff.nagl.molecule._graph._utils import _batch_nx_graphs

        if not molecules:
            raise ValueError("No molecules were provided.")
        batched_graph = molecules[0].graph._batch([molecule.graph for molecule in molecules])
        n_representations = tuple(molecule.n_representations for molecule in molecules)
        n_atoms = tuple(molecule.n_atoms for molecule in molecules)
        return cls(graph=batched_graph, n_representations=n_representations, n_atoms=n_atoms)