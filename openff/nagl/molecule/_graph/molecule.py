from typing import List, TYPE_CHECKING, Tuple, Optional

from openff.nagl.molecule._base import NAGLMoleculeBase, MoleculeMixin, BatchMixin
from openff.nagl.molecule._graph._graph import NXMolHeteroGraph

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule
    from openff.nagl.features.atoms import AtomFeature
    from openff.nagl.features.bonds import BondFeature


class GraphMolecule(MoleculeMixin, NAGLMoleculeBase):
    def to_homogenous(self):
        return self.graph.to_homogeneous()

    def to(self, device: str):
        return type(self)(self.graph, n_representations=self.n_representations)

    @property
    def n_graph_nodes(self):
        return int(self.graph.graph.number_of_nodes())

    @property
    def n_graph_edges(self):
        return int(self.graph.graph.number_of_edges())

    @classmethod
    def from_openff(
        cls,
        molecule: "Molecule",
        atom_features: Tuple["AtomFeature", ...] = tuple(),
        bond_features: Tuple["BondFeature", ...] = tuple(),
        enumerate_resonance_forms: bool = False,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
    ):
        from openff.nagl.utils.resonance import ResonanceEnumerator

        offmols = [molecule]
        if enumerate_resonance_forms:
            enumerator = ResonanceEnumerator(molecule)
            offmols = enumerator.enumerate_resonance_forms(
                lowest_energy_only=lowest_energy_only,
                max_path_length=max_path_length,
                include_all_transfer_pathways=include_all_transfer_pathways,
                as_dicts=False,
            )
        graphs = [
            NXMolHeteroGraph.from_openff(
                offmol,
                atom_features=atom_features,
                bond_features=bond_features,
            )
            for offmol in offmols
        ]
        graph = NXMolHeteroGraph._batch(graphs)

        mapped_smiles = molecule.to_smiles(mapped=True)

        return cls(
            graph=graph,
            n_representations=len(graphs),
            mapped_smiles=mapped_smiles,
        )


class GraphMoleculeBatch(BatchMixin, NAGLMoleculeBase):
    def to(self, device: str):
        return type(self)(
            self.graph, n_representations=self.n_representations, n_atoms=self.n_atoms
        )

    @classmethod
    def from_nx_molecules(cls, molecules: List[GraphMolecule]):
        if not molecules:
            raise ValueError("No molecules were provided.")
        batched_graph = molecules[0].graph._batch(
            [molecule.graph for molecule in molecules]
        )
        batched_graph.batch_size = len(molecules)
        n_representations = tuple(molecule.n_representations for molecule in molecules)
        n_atoms = tuple(molecule.n_atoms for molecule in molecules)
        return cls(
            graph=batched_graph, n_representations=n_representations, n_atoms=n_atoms
        )

    def unbatch(self) -> List[GraphMolecule]:

        return [
            GraphMolecule(g, n_repr)
            for g, n_repr in zip(
                self.graph.unbatch(self.n_representations_per_molecule),
                self.n_representations_per_molecule
            )
        ]