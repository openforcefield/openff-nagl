from typing import ClassVar, Optional, Tuple

import torch
from openff.toolkit.topology.molecule import Molecule as OFFMolecule
from openff.utilities import requires_package

from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.molecule._base import NAGLMoleculeBase, MoleculeMixin
from openff.nagl.toolkits.openff import capture_toolkit_warnings
from .utils import (
    FORWARD,
    dgl_heterograph_to_homograph,
    openff_molecule_to_dgl_graph,
)


class DGLBase(NAGLMoleculeBase):
    def to_homogenous(self):
        return dgl_heterograph_to_homograph(self.graph)

    def to(self, device: str):
        graph = self.graph.to(device)
        return type(self)(graph)


class DGLMolecule(MoleculeMixin, DGLBase):
    n_representations: int = 1

    def to(self, device: str):
        graph = self.graph.to(device)
        return type(self)(graph, n_representations=self.n_representations)

    @property
    def n_graph_nodes(self):
        return int(self.graph.number_of_nodes())

    @property
    def n_graph_edges(self):
        return int(self.graph.number_of_edges(FORWARD))

    @classmethod
    @requires_package("dgl")
    def from_openff(
        cls,
        molecule: OFFMolecule,
        atom_features: Tuple[AtomFeature, ...] = tuple(),
        bond_features: Tuple[BondFeature, ...] = tuple(),
        atom_feature_tensor: Optional[torch.Tensor] = None,
        bond_feature_tensor: Optional[torch.Tensor] = None,
        enumerate_resonance_forms: bool = False,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
    ):
        import dgl
        from openff.nagl.utils.resonance import ResonanceEnumerator

        if len(atom_features) and atom_feature_tensor is not None:
            raise ValueError(
                "Only one of `atom_features` or "
                "`atom_feature_tensor` should be provided."
            )

        if len(bond_features) and bond_feature_tensor is not None:
            raise ValueError(
                "Only one of `bond_features` or "
                "`bond_feature_tensor` should be provided."
            )

        offmols = [molecule]
        if enumerate_resonance_forms:
            enumerator = ResonanceEnumerator(molecule)
            offmols = enumerator.enumerate_resonance_forms(
                lowest_energy_only=lowest_energy_only,
                max_path_length=max_path_length,
                include_all_transfer_pathways=include_all_transfer_pathways,
                as_dicts=False,
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

        with capture_toolkit_warnings():
            mapped_smiles = offmols[0].to_smiles(mapped=True)

        return cls(
            graph=graph,
            n_representations=len(offmols),
            mapped_smiles=mapped_smiles
        )
