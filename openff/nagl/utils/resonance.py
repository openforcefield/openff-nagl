

import copy
import itertools
import json
from typing import Dict, Optional, List, Generator, Tuple, Any

import networkx as nx
import numpy as np

from openff.toolkit.topology import Molecule

from openff.nagl.utils.types import ResonanceType, ResonanceAtomType

__all__ = ["ResonanceEnumerator"]


class ResonanceEnumerator:


    def __init__(
        self,
        molecule: Molecule
    ):
        self.molecule = molecule
        self.graph = self._convert_molecule_to_graph(molecule)
        self.reduced_graph = self._reduce_graph(self.graph, inplace=False)

    def enumerate_resonance_forms(
        self,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
        as_dicts: bool = False,
    ):
        fragments = self._enumerate_resonance_fragments(
            lowest_energy_only=lowest_energy_only,
            max_path_length=max_path_length,
            include_all_transfer_pathways=include_all_transfer_pathways,
        )
        combinations = itertools.product(*fragments)
        resonance_forms = [
            self._substitute_resonance_fragments(combination)
            for combination in combinations
        ]

        if as_dicts:
            molecules = [
                self._convert_graph_to_dict(resonance_form)
                for resonance_form in resonance_forms
            ]
        else:
            molecules = [
                Molecule.from_networkx(resonance_form)
                for resonance_form in resonance_forms
            ]
        
        return molecules


    @staticmethod
    def _convert_graph_to_dict(graph: nx.Graph) -> Dict[str, Dict[str, Any]]:
        atoms = dict(graph.nodes(data=True))
        bonds = {}
        for i, j, info in graph.edges(data=True):
            if j < i:
                i, j = j, i
            bonds[(i, j)] = info
        return {"atoms": atoms, "bonds": bonds}

    def _enumerate_resonance_fragments(
        self,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
    ) -> List[List[nx.Graph]]:
        acceptor_donor_fragments = self.get_acceptor_donor_fragments()
        fragment_resonance_forms = [
            fragment._enumerate_resonance_forms(
                lowest_energy_only=lowest_energy_only,
                max_path_length=max_path_length,
                include_all_transfer_pathways=include_all_transfer_pathways,
            )
            for fragment in acceptor_donor_fragments
        ]
        return fragment_resonance_forms
    
    def _substitute_resonance_fragments(self, resonance_forms: List[nx.Graph]):
        graph = copy.deepcopy(self.graph)
        for subgraph in resonance_forms:
            self._substitute_graph_attributes(subgraph, graph)
        return graph
    

    @staticmethod
    def _substitute_graph_attributes(source: nx.Graph, target: nx.Graph):
        for node in target.nodes:
            target.nodes[node].update(source.nodes[node])
        for i, j in target.edges:
            target.edges[i][j].update(source.edges[i][j])


    @staticmethod
    def _convert_molecule_to_graph(molecule):
        graph = molecule.to_networkx()
        for node, atom in zip(graph.nodes, molecule.atoms):
            bond_orders = tuple(sorted(bond.bond_order for bond in atom.bonds))
            graph.nodes[node]["bond_orders"] = tuple(bond_orders)
    
        return graph

    @staticmethod
    def _reduce_graph(graph, inplace: bool = True):
        if not inplace:
            graph = copy.deepcopy(graph)

        nodes_to_remove = []
        for index in graph.nodes:
            atom_info = graph.nodes[index]

            # remove hydrogens. These are implicitly captured by bond_orders
            if atom_info["atomic_number"] == 1:
                nodes_to_remove.append(index)
            
            # Next prune all CX4 carbon atoms - because they have no double bonds to begin with
            # there is no electron transfer that can occur to change that  otherwise they will
            # end up pentavalent, and so can never be part of a conjugated path

            elif (
                atom_info["atomic_number"] == 6
                and atom_info["bond_orders"] == (1, 1, 1, 1)
                and atom_info["formal_charge"] == 0
            ):
                nodes_to_remove.append(index)
        
        graph.remove_nodes_from(nodes_to_remove)
        return graph

    @staticmethod
    def _fragment_networkx_graph(graph):
        return [
            graph.subgraph(fragment)
            for fragment in nx.connected_components(graph)
        ]
    
    def get_acceptor_donor_fragments(self):
        acceptor_donor_fragments = []
        
        for nxfragment in self._fragment_networkx_graph(self.reduced_graph):
            fragment = FragmentEnumerator(nxfragment)
            if fragment.acceptor_indices and fragment.donor_indices:
                acceptor_donor_fragments.append(fragment)
        
        return acceptor_donor_fragments
    


    


class FragmentEnumerator:
    def __init__(self, graph):
        self.graph = graph
        self._path_cache = {}
        self.resonance_types = self._get_resonance_types()
        self.acceptor_indices = []
        self.donor_indices = []

        for index, resonance_type in self.resonance_types.items():
            if resonance_type.type == ResonanceAtomType.Acceptor.value:
                self.acceptor_indices.append(index)
            elif resonance_type.type == ResonanceAtomType.Donor.value:
                self.donor_indices.append(index)
        

    def _get_all_odd_n_simple_paths(
        self,
        node_a: int,
        node_b: int,
        max_path_length: Optional[int] = None,
    ) -> Tuple[Tuple[int, ...], ...]:
        """
        Get all odd length simple paths between two nodes in a graph.


        Parameters
        ----------
        node_a: int
            The index of the first node.
        node_b: int
            The index of the second node.
        max_path_length: Optional[int]
            The maximum length of the paths to return. If None, all paths will be returned.
        
        Returns
        -------
        all_odd_paths: Tuple[Tuple[int, ...], ...]
            All odd length simple paths between the two nodes.
        """
        key = (node_a, node_b)
        if key in self._path_cache:
            return self._path_cache[key]
        
        all_paths = map(
            tuple,
            nx.all_simple_paths(
                self.graph,
                node_a,
                node_b,
                cutoff=max_path_length,
            )
        )
        odd_paths = tuple([path for path in all_paths if len(path) % 2 == 1])
        self._path_cache[key] = odd_paths
        self._path_cache[key[::-1]] = odd_paths
        return odd_paths

    def _get_transfer_paths(
        self,
        donor_node: int,
        acceptor_node: int,
        max_path_length: Optional[int] = None,
    ) -> Generator[Tuple[int, ...], None, None]:
        for path in self._get_all_odd_n_simple_paths(donor_node, acceptor_node, max_path_length):
            if self._is_transfer_path(path):
                yield path


    def _is_transfer_path(self, path: Tuple[int, ...]) -> bool:
        """
        Determine if a path is a transfer path based on alternating bond order.
        This checks whether the bonds along the path form a rising-falling sequence.

        Parameters
        ----------
        path: Tuple[int, ...]
            The path to check. Each number is a node in the graph that
            should be bonded to the nodes on either side.

        Returns
        -------
        is_transfer_path: bool
            Whether the path is a transfer path.
        """
        edges = zip(path[:-1], path[1:])
        bond_orders = [self.graph.edges[i, j]["bond_order"] for i, j in edges]
        deltas = bond_orders[1:] - bond_orders[:-1]
        return np.all(deltas[::2] == 1) and np.all(deltas[1::2] == -1)


    def _to_resonance_dict(
        self,
        include_bonds: bool = True,
        include_formal_charges: bool = False
    ) -> Dict[str, List[int]]:
        hash_dict = {
            "acceptor_indices": self.acceptor_indices,
            "donor_indices": self.donor_indices,
        }
        if include_bonds:
            hash_dict["bond_orders"] = sorted(
                self.graph.edges(data="bond_order")
            )

        if include_formal_charges:
            hash_dict["formal_charges"] = sorted(
                self.graph.nodes(data="formal_charge")
            )
        return hash_dict

    def _to_resonance_json(
        self,
        include_bonds: bool = True,
        include_formal_charges: bool = False
    ) -> str:
        return json.dumps(
            self._to_resonance_dict(
                include_bonds=include_bonds,
                include_formal_charges=include_formal_charges,
            ),
            sort_keys=True,
        )


    def _to_resonance_hash(
        self,
        include_bonds: bool = True,
        include_formal_charges: bool = False
    ) -> bytes:
        import hashlib

        json_str = self.to_resonance_json(
            include_bonds=include_bonds, 
            nclude_formal_charges=include_formal_charges
        )
        return hashlib.sha1(json_str.encode(), usedforsecurity=False).digest()


    def _transfer_electrons(self, path: Tuple[int, ...]):
        graph = copy.deepcopy(self.graph)

        donor_index, acceptor_index = path[0], path[-1]
        for index in [donor_index, acceptor_index]:
            resonance_type = self._get_atom_resonance_type(index)
            conjugate_key = resonance_type.get_conjugate_key()
            graph.nodes[index]["atomic_number"] = conjugate_key.atomic_number
            graph.nodes[index]["formal_charge"] = conjugate_key.formal_charge
            graph.nodes[index]["bond_orders"] = conjugate_key.bond_orders
        
        for bond_index, (i, j) in enumerate(zip(path[:-1], path[1:])):
            increment = 1 if bond_index % 2 == 0 else -1
            graph.edges[i, j]["bond_order"] += increment
        
        return type(self)(graph)



    

    def _get_atom_resonance_type(self, index: int) -> ResonanceType.Value:
        """Get the resonance type of an atom."""
        node_info = self.graph[index]
        return ResonanceType.get_resonance_type(
            atomic_number=node_info["atomic_number"],
            formal_charge=node_info["formal_charge"],
            bond_orders=node_info["bond_orders"],
        )

    def _get_resonance_types(self) -> Dict[int, ResonanceType.Value]:
        """Get resonance types of acceptor and donor atoms.
        
        Returns
        -------
        resonance_types: Dict[int, ResonanceType.Value]
            A dictionary of resonance types for each atom in the graph.
            Keys are node numbers.
        """
        resonance_types = {}
        for index in self.graph.nodes:
            try:
                resonance_types[index] = self._get_atom_resonance_type(index)
            except KeyError:
                pass
        return resonance_types

    def _enumerate_donor_acceptor_resonance_forms(
        self,
        max_path_length: Optional[int] = None,
    ) -> Generator["FragmentEnumerator", None, None]:
        for acceptor_index in self.acceptor_indices:
            for donor_index in self.donor_indices:
                for path in self._get_transfer_paths(
                    donor_index,
                    acceptor_index,
                    max_path_length
                ):
                    transferred = self._transfer_electrons(path)
                    yield transferred


    def enumerate_resonance_forms(
        self,
        include_all_transfer_pathways: bool = False,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
    ) -> List[nx.Graph]:
        """
        Enumerate all resonance forms for a fragment.

        Parameters
        ----------
        include_all_transfer_pathways: bool
            Whether to include all possible transfer pathways.
            If False, drop resonance forms that only differ due to bond order
            rearrangements, e.g. ring resonance structures
        lowest_energy_only: bool
            Whether to only include the lowest energy resonance forms.
        """
        
        self_hash = self._to_resonance_hash(include_bonds=True)
        open_forms: Dict[bytes, FragmentEnumerator] = {self_hash: self}
        closed_forms: Dict[bytes, FragmentEnumerator] = {}

        while open_forms:
            current_forms: Dict[bytes, FragmentEnumerator] = {}

            for current_key, current_fragment in open_forms.items():
                if current_key in closed_forms:
                    continue
                closed_forms[current_key] = current_fragment
            
                new_forms = current_fragment._enumerate_donor_acceptor_resonance_forms(max_path_length=max_path_length)
                for new_fragment in new_forms:
                    new_key = new_fragment._to_resonance_hash(include_bonds=True)
                    if new_key not in closed_forms and new_key not in open_forms:
                        current_forms[new_key] = new_fragment

            open_forms = current_forms
        
        if not include_all_transfer_pathways:
            # Drop resonance forms that only differ due to bond order re-arrangements as we
            # aren't interested in e.g. ring resonance structures.
            closed_forms = {
                fragment.to_resonance_hash(include_bonds=False): fragment
                for fragment in closed_forms.values()
            }

        if lowest_energy_only:
            closed_forms = self._select_lowest_energy_forms(closed_forms)

        graphs = [form.graph for form in closed_forms.values()]

        return graphs

    
    @staticmethod
    def _select_lowest_energy_forms(
        forms: Dict[Any, "FragmentEnumerator"]
    ) -> Dict[Any, "FragmentEnumerator"]:
        """Select the resonance forms with the lowest energy."""

        energies: Dict[Any, float] = {
            key: form._calculate_resonance_energy()
            for key, form in forms.items()
        }
        lowest = min(energies.values())
        lowest_forms = {
            key: forms[key]
            for key, energy in energies.items()
            if np.isclose(energy, lowest)
        }
        return lowest_forms


    def _get_resonance_energy(self) -> float:
        return sum(res.energy for res in self.resonance_types.values())