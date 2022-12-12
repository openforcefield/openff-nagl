from typing import ClassVar, Optional, Tuple

import dgl.function
import torch
from openff.toolkit.topology.molecule import Molecule as OFFMolecule

from ..base.base import ImmutableModel
from ..features.atoms import AtomFeature
from ..features.bonds import BondFeature
from ..resonance.resonance import ResonanceEnumerator
from .utils import (
    FEATURE,
    FORWARD,
    dgl_heterograph_to_homograph,
    openff_molecule_to_dgl_graph,
)

__all__ = [
    "DGLBase",
    "DGLMolecule",
]


class DGLBase(ImmutableModel):
    graph: dgl.DGLHeteroGraph

    _graph_feature_name: ClassVar[str] = "h"
    _graph_forward_edge_type: ClassVar[str] = "forward"
    _graph_backward_edge_type: ClassVar[str] = "reverse"

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
    def from_openff(
        cls,
        molecule: OFFMolecule,
        atom_features: Tuple[AtomFeature] = tuple(),
        bond_features: Tuple[BondFeature] = tuple(),
        enumerate_resonance_forms: bool = False,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
    ):
        offmols = [molecule]
        if enumerate_resonance_forms:
            enumerator = ResonanceEnumerator(molecule)
            offmols = enumerator.enumerate_resonance_molecules(
                lowest_energy_only=lowest_energy_only,
                max_path_length=max_path_length,
                include_all_transfer_pathways=include_all_transfer_pathways,
                moleculetype=OFFMolecule,
            )

        subgraphs = [
            openff_molecule_to_dgl_graph(
                offmol,
                atom_features,
                bond_features,
                forward=cls._graph_forward_edge_type,
                reverse=cls._graph_backward_edge_type,
            )
            for offmol in offmols
        ]
        graph = dgl.batch(subgraphs)
        graph.set_batch_num_nodes(graph.batch_num_nodes().sum().reshape((-1,)))
        graph.set_batch_num_edges(
            {
                e_type: graph.batch_num_edges(e_type).sum().reshape((-1,))
                for e_type in graph.canonical_etypes
            }
        )

        return cls(graph=graph, n_representations=len(offmols))

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        mapped: bool = False,
        atom_features: Tuple[AtomFeature] = tuple(),
        bond_features: Tuple[BondFeature] = tuple(),
        enumerate_resonance_forms: bool = False,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
    ):
        func = OFFMolecule.from_smiles
        if mapped:
            func = OFFMolecule.from_mapped_smiles
        molecule = func(smiles)
        return cls.from_openff(
            molecule=molecule,
            atom_features=atom_features,
            bond_features=bond_features,
            enumerate_resonance_forms=enumerate_resonance_forms,
            lowest_energy_only=lowest_energy_only,
            max_path_length=max_path_length,
            include_all_transfer_pathways=include_all_transfer_pathways,
        )
