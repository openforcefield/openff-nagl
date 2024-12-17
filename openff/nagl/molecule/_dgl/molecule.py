from typing import ClassVar, Optional, Tuple, TYPE_CHECKING

import torch
from openff.utilities import requires_package

from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.molecule._base import NAGLMoleculeBase, MoleculeMixin
from .utils import (
    FORWARD, REVERSE,
    dgl_heterograph_to_homograph,
    openff_molecule_to_dgl_graph,
)

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


class DGLBase(NAGLMoleculeBase):
    def to_homogenous(self):
        return dgl_heterograph_to_homograph(self.graph)

    def to(self, device: str):
        graph = self.graph.to(device)
        return type(self)(graph)
    
    def _get_bonds(self) -> list[tuple[int, int]]:
        a, b = self.graph.edges(etype=FORWARD)
        bonds = list(zip(a.tolist(), b.tolist()))
        # sort
        bonds = [tuple(sorted(bond)) for bond in bonds]
        return sorted(bonds)
    
    def _get_angles(self) -> list[tuple[int, int, int]]:
        angles = set()
        bonds = self._get_bonds()
        bonds += [(b, a) for a, b in bonds]
        for atom1, atom2 in bonds:
            atom3s = self.graph.successors(atom2, etype=FORWARD).tolist()
            # TODO: necessary?
            atom3s += self.graph.predecessors(atom2, etype=FORWARD).tolist()
            for atom3 in atom3s:
                if atom1 == atom3:
                    continue
                if atom1 < atom3:
                    angles.add((atom1, atom2, atom3))
                else:
                    angles.add((atom3, atom2, atom1))
        return sorted(list(angles))

    def _get_dihedrals(self) -> list[tuple[int, int, int, int]]:
        dihedrals = set()
        angles = self._get_angles()
        angles += [(c, b, a) for a, b, c in angles]
        for atom1, atom2, atom3 in angles:
            atom4s = self.graph.successors(atom3, etype=FORWARD).tolist()
            # TODO: necessary?
            atom4s += self.graph.predecessors(atom3, etype=FORWARD).tolist()
            for atom4 in atom4s:
                if atom2 == atom4:
                    continue
                if atom1 < atom4:
                    dihedrals.add((atom1, atom2, atom3, atom4))
                else:
                    dihedrals.add((atom4, atom3, atom2, atom1))
            
            atom0s = self.graph.successors(atom1, etype=FORWARD).tolist()
            atom0s += self.graph.predecessors(atom1, etype=FORWARD).tolist()
            for atom0 in atom0s:
                if atom2 == atom0:
                    continue
                if atom3 < atom0:
                    dihedrals.add((atom3, atom2, atom1, atom0))
                else:
                    dihedrals.add((atom0, atom1, atom2, atom3))
        return sorted(list(dihedrals))




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
    def from_openff_config(
        cls,
        molecule,
        model_config,
        atom_feature_tensor: Optional[torch.Tensor] = None,
        bond_feature_tensor: Optional[torch.Tensor] = None,
        model=None,
    ):
        return cls.from_openff(
            molecule,
            atom_features=model_config.atom_features,
            bond_features=model_config.bond_features,
            atom_feature_tensor=atom_feature_tensor,
            bond_feature_tensor=bond_feature_tensor,
            enumerate_resonance_forms=model_config.enumerate_resonance_forms,
            lowest_energy_only=True,
            max_path_length=None,
            include_all_transfer_pathways=False,
            include_xyz=model_config.include_xyz,
            model=model,
        )

    @classmethod
    @requires_package("dgl")
    def from_openff(
        cls,
        molecule: "Molecule",
        atom_features: Tuple[AtomFeature, ...] = tuple(),
        bond_features: Tuple[BondFeature, ...] = tuple(),
        atom_feature_tensor: Optional[torch.Tensor] = None,
        bond_feature_tensor: Optional[torch.Tensor] = None,
        enumerate_resonance_forms: bool = False,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
        include_xyz: bool = False,
        model=None,
    ):
        import dgl
        from openff.nagl.utils.resonance import ResonanceEnumerator

        if atom_features is None:
            atom_features = tuple()
        if bond_features is None:
            bond_features = tuple()

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
                atom_features=atom_features,
                bond_features=bond_features,
                atom_feature_tensor=atom_feature_tensor,
                bond_feature_tensor=bond_feature_tensor,
                forward=cls._graph_forward_edge_type,
                reverse=cls._graph_backward_edge_type,
                include_xyz=include_xyz
            )
            for offmol in offmols
        ]
        graph = dgl.batch(subgraphs)
        n_nodes = graph.batch_num_nodes().sum().reshape((-1,))
        graph.set_batch_num_nodes(n_nodes.type(torch.int32))
        edges = {}
        for e_type in graph.canonical_etypes:
            n_edge = graph.batch_num_edges(e_type).sum().reshape((-1,))
            edges[e_type] = n_edge.type(torch.int32)
        graph.set_batch_num_edges(edges)

        mapped_smiles = offmols[0].to_smiles(mapped=True)

        obj = cls(
            graph=graph,
            n_representations=len(offmols),
            mapped_smiles=mapped_smiles
        )
        # n_atoms = len(offmols[0].atoms)
        # if model is not None:
        #     all_pooling_layers = [
        #         readout.pooling_layer
        #         for readout in model.readout_modules.values()
        #     ]
        #     for pooling_layer in all_pooling_layers:
        #         if pooling_layer.name == "atom":
        #             continue
        #         indices = pooling_layer._generate_transposed_pooling_representation(
        #             molecule
        #         )
        #         all_indices = []
        #         for i in range(len(offmols)):
        #             all_indices.append(indices + (i * n_atoms))
        #         indices = torch.cat(all_indices, dim=1)
        #         obj._pooling_representations[pooling_layer.name] = indices
        return obj
