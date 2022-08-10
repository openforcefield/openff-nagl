import copy
from typing import List

import dgl.function
import torch

from openff.toolkit.topology.molecule import (
    Molecule as OFFMolecule,
    unit as off_unit,
)

from ..base.base import ImmutableModel
from ..features.atoms import AtomFeature
from ..features.bonds import BondFeature
from .utils import FORWARD, FEATURE, openff_molecule_to_dgl_graph, dgl_heterograph_to_homograph


class DGLBase(ImmutableModel):
    graph: dgl.DGLHeteroGraph

    @property
    def atom_features(self) -> torch.Tensor:
        return self.graph.ndata[FEATURE].float()

    @property
    def homograph(self):
        return self.to_homogenous()

    def to_homogenous(self):
        return dgl_heterograph_to_homograph(self.graph)

    def to(self, device: str):
        copied = self.copy(deep=False, update={"graph": self.graph.to(device)})
        return copied


class DGLMolecule(DGLBase):
    n_representations: int = 1

    @property
    def n_graph_nodes(self):
        return int(self.graph.number_of_nodes())

    @property
    def n_graph_edges(self):
        return int(self.graph.number_of_edges(FORWARD))

    @property
    def n_atoms(self):
        return self.n_graph_nodes / self.n_representations

    @property
    def n_atoms_per_molecule(self):
        return (self.n_atoms,)

    @property
    def n_bonds(self):
        return self.n_graph_edges / self.n_representations

    @property
    def n_representations_per_molecule(self):
        return (self.n_representations,)

    @classmethod
    def openff_molecule_to_dgl_graph(
        cls,
        molecule: OFFMolecule,
        atom_features: List[AtomFeature] = [],
        bond_features: List[BondFeature] = [],
    ) -> dgl.DGLGraph:
        return openff_molecule_to_dgl_graph(
            molecule, atom_features, bond_features
        )

    @classmethod
    def from_openff(
        cls,
        molecule: OFFMolecule,
        atom_features: List[AtomFeature] = [],
        bond_features: List[BondFeature] = [],
    ):
        graph = cls.openff_molecule_to_dgl_graph(
            molecule=molecule,
            atom_features=atom_features,
            bond_features=bond_features,
        )
        return cls(graph=graph, n_representations=1)

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        atom_features: List[AtomFeature] = [],
        bond_features: List[BondFeature] = [],
    ):
        molecule = OFFMolecule.from_smiles(smiles)
        return cls.from_openff(
            molecule=molecule,
            atom_features=atom_features,
            bond_features=bond_features,
        )
