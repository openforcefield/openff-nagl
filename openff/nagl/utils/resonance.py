"""Enumerate resonance forms of a molecule"""

import copy
import itertools
import json
from typing import Dict, Optional, List, Generator, Tuple, Any, Union

import networkx as nx
import numpy as np

from openff.units import unit
from openff.toolkit.topology import Molecule

from openff.nagl.utils._types import ResonanceType, ResonanceAtomType
from openff.nagl.toolkits.openff import _molecule_from_dict, _molecule_to_dict

__all__ = ["ResonanceEnumerator", "enumerate_resonance_forms"]


def enumerate_resonance_forms(
    molecule: Molecule,
    lowest_energy_only: bool = True,
    max_path_length: Optional[int] = None,
    include_all_transfer_pathways: bool = False,
    as_dicts: bool = False,
    as_fragments: bool = False,
) -> List[Union[Molecule, Dict[str, Dict[str, Any]]]]:
    """
    Find all resonance structures of ``molecule`` according to Gilson et al [1].

    Recursively attempts to find all resonance structures of an input molecule
    according to a modified version of the algorithm proposed by Gilson et al [1].
    Enumeration proceeds by:

    1. The molecule is turned into a ``networkx`` graph object.
    2. All hydrogen's and uncharged sp3 carbons are removed from the graph as these
       will not be involved in electron transfer.
    3. Disjoint sub-graphs are detected and separated out.
    4. Sub-graphs that don't contain at least 1 donor and 1 acceptor are discarded
    5. For each disjoint subgraph:
        a) The original v-charge algorithm is applied to yield the resonance structures
        of that subgraph.

    This will lead to ``M_i`` resonance structures for each of the ``N`` sub-graphs.

    If ``as_dicts=True`` then the resonance states in each sub-graph are returned. This
    avoids the need to combinatorially combining resonance information from each
    sub-graph. When ``as_dicts=False``, all ``M_0 x M_1 x ... x M_N`` forms are fully
    enumerated and return as molecule objects matching the input molecule type.

    .. note::

        * This method will strip all stereochemistry and aromaticity information from
          the input molecule.
        * The method only attempts to enumerate resonance forms that occur when a
          pair of electrons can be transferred along a conjugated path from a donor to
          an acceptor. Other types of resonance, e.g. different Kekule structures, are
          not enumerated.

    Parameters
    ----------
    molecule: openff.toolkit.topology.Molecule
        The molecule to enumerate resonance forms for.
    lowest_energy_only: bool, optional
        Whether to only return the resonance forms with the lowest
        'energy' as defined in [1].
    max_path_length: int, optional
        The maximum number of bonds between a donor and acceptor to
        consider. If None, all paths are considered.
    as_dicts: bool, optional
        Whether to return the resonance forms in a form that is more
        compatible with producing feature vectors. If false, all combinatorial
        resonance forms will be returned which may be significantly slow if the
        molecule is very heavily conjugated and has many donor / acceptor pairs.
    include_all_transfer_pathways: bool, optional
        Whether to include resonance forms that have
        the same formal charges but have different arrangements of bond orders. Such
        cases occur when there exists multiple electron transfer pathways between
        electron donor-acceptor pairs e.g. in cyclic systems.

    References
    ----------

    [1] Gilson, Michael K., Hillary SR Gilson, and Michael J. Potter. "Fast
    assignment of accurate partial atomic charges: an electronegativity
    equalization method that accounts for alternate resonance forms." Journal of
    chemical information and computer sciences 43.6 (2003): 1982-1997.

    Returns
    -------
    resonance_forms
        A list of all resonance forms including the original molecule.

    """
    enumerator = ResonanceEnumerator(molecule)
    return enumerator.enumerate_resonance_forms(
        lowest_energy_only=lowest_energy_only,
        max_path_length=max_path_length,
        include_all_transfer_pathways=include_all_transfer_pathways,
        as_dicts=as_dicts,
        as_fragments=as_fragments
    )


class ResonanceEnumerator:
    """
    A convenience class for enumerating resonance forms of a molecule
    according to the algorithm proposed by Gilson et al [1].

    Enumeration proceeds by:

    1. The molecule is turned into a ``networkx`` graph object at :attr:`ResonanceEnumerator.graph`.
    2. All hydrogen's and uncharged sp3 carbons are removed from the graph as these
       will not be involved in electron transfer, yielding :attr:`ResonanceEnumerator.reduced_graph`.
    3. Disjoint sub-graphs are detected and separated out.
    4. Sub-graphs that don't contain at least 1 donor and 1 acceptor are discarded
    5. For each disjoint subgraph:
        a) The ``networkx`` graph is converted to a :class:`FragmentEnumerator`
        b) The original v-charge algorithm is applied to yield the resonance structures
           of that subgraph, using :meth:`FragmentEnumerator.enumerate_resonance_forms`

    """

    def __init__(self, molecule: Molecule):
        self.molecule = molecule
        self.graph = self._convert_molecule_to_graph(molecule)
        self._graph_dict = _molecule_to_dict(molecule)
        self.reduced_graph = self._reduce_graph(self.graph, inplace=False)

    def enumerate_resonance_forms(
        self,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
        as_dicts: bool = False,
        as_fragments: bool = False,
    ) -> List[Union[Molecule, Dict[str, Dict[str, Any]]]]:
        """
        Recursively attempts to find all resonance structures of an input molecule
        according to a modified version of the algorithm proposed by Gilson et al [1].

        Enumeration proceeds by:

        1. The molecule is turned into a ``networkx`` graph object.
        2. All hydrogen's and uncharged sp3 carbons are removed from the graph as these
           will not be involved in electron transfer.
        3. Disjoint sub-graphs are detected and separated out.
        4. Sub-graphs that don't contain at least 1 donor and 1 acceptor are discarded
        5. For each disjoint subgraph:
            a) The original v-charge algorithm is applied to yield the resonance structures
               of that subgraph.

        Parameters
        ----------
        lowest_energy_only: bool, optional
            Whether to only return the resonance forms with the lowest
            'energy' as defined in [1].
        max_path_length: int, optional
            The maximum number of bonds between a donor and acceptor to
            consider. If None, all paths are considered.
        as_dicts: bool, optional
            Whether to return the resonance forms as dictionaries.
            If ``False``, all combinatorial
            resonance forms will be returned as molecule objects.
            This may be significantly slow if the
            molecule is very heavily conjugated and has many donor / acceptor pairs.
            If ``True`` then the resonance states in each sub-graph are returned. This
            avoids the need to combinatorially combining resonance information from each
            sub-graph.
        include_all_transfer_pathways: bool, optional
            Whether to include resonance forms that have
            the same formal charges but have different arrangements of bond orders. Such
            cases occur when there exists multiple electron transfer pathways between
            electron donor-acceptor pairs e.g. in cyclic systems.

        References
        ----------

        [1] Gilson, Michael K., Hillary SR Gilson, and Michael J. Potter. "Fast
        assignment of accurate partial atomic charges: an electronegativity
        equalization method that accounts for alternate resonance forms." Journal of
        chemical information and computer sciences 43.6 (2003): 1982-1997.

        Returns
        -------
        resonance_forms
            A list of all resonance forms including the original molecule.

        """
        from openff.nagl.toolkits.openff import molecule_from_networkx

        all_fragments = self._enumerate_resonance_fragments(
            lowest_energy_only=lowest_energy_only,
            max_path_length=max_path_length,
            include_all_transfer_pathways=include_all_transfer_pathways,
        )
        graphs = [
            [fragment.reduced_graph for fragment in fragments]
            for fragments in all_fragments
        ]
        
        if not as_fragments:
            combinations = itertools.product(*graphs)
            resonance_forms = [
                self._substitute_resonance_fragments(combination)
                for combination in combinations
            ]

            if as_dicts:
                # molecules = [
                #     self._convert_graph_to_dict(resonance_form)
                #     for resonance_form in resonance_forms
                # ]
                molecules = resonance_forms
            else:
                molecules = [
                    # molecule_from_networkx(resonance_form)
                    _molecule_from_dict(resonance_form)
                    for resonance_form in resonance_forms
                ]
        
        else:
            if not as_dicts:
                raise NotImplementedError("as_fragments=True requires as_dicts=True")
            
            molecules = []
            for fragments in graphs:
                for subgraph in fragments:
                    atoms = {
                        node: subgraph.nodes[node]
                        for node in subgraph.nodes
                    }
                    bonds = {}
                    for i, j in subgraph.edges:
                        key = tuple(sorted((i, j)))
                        bonds[key] = subgraph.edges[i, j]
                    molecules.append({"atoms": atoms, "bonds": bonds})


        return molecules

    def to_fragment(self) -> "FragmentEnumerator":
        """Convert to a FragmentEnumerator"""
        graph = copy.deepcopy(self.reduced_graph)
        return FragmentEnumerator(graph)

    @staticmethod
    def _convert_graph_to_dict(graph: nx.Graph) -> Dict[str, Dict[str, Any]]:
        """
        Convert a molecule ``networkx`` graph to a dictionary

        Parameters
        ----------
        graph: :class:`networkx.Graph`
            Input molecule graph

        Returns
        -------
        molecule_dict: Dict[str, Dict[str, Any]]
            A dictionary representation of the molecule.
            molecule_dict["atoms"][i] contains the atom information for atom i.
            molecule_dict["bonds"][(i, j)] contains the bond information for bond (i, j).
            All bonds are stored in the order (i, j) where i < j.
        """
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
    ) -> List[List["FragmentEnumerator"]]:
        """
        Recursively enumerate all resonance forms of disjoint subgraphs.

        Parameters
        ----------
        lowest_energy_only: bool, optional
            Whether to only return the resonance forms with the lowest
            'energy' as defined in [1].
        max_path_length: int, optional
            The maximum number of bonds between a donor and acceptor to
            consider. If None, all paths are considered.
        include_all_transfer_pathways: bool, optional
            Whether to include resonance forms that have
            the same formal charges but have different arrangements of bond orders. Such
            cases occur when there exists multiple electron transfer pathways between
            electron donor-acceptor pairs e.g. in cyclic systems.

        Returns
        -------
        resonance_fragments: List[List[FragmentEnumerator]]
            All enumerated disjoint subgraphs.
            resonance_fragments[i][j] returns the j-th resonance form
            of the i-th subgraph of the molecule.
        """
        acceptor_donor_fragments = self._get_acceptor_donor_fragments()
        fragment_resonance_forms = [
            fragment.enumerate_resonance_forms(
                lowest_energy_only=lowest_energy_only,
                max_path_length=max_path_length,
                include_all_transfer_pathways=include_all_transfer_pathways,
            )
            for fragment in acceptor_donor_fragments
        ]
        return fragment_resonance_forms

    def _substitute_resonance_fragments(
        self, resonance_forms: List[nx.Graph]
    ) -> nx.Graph:
        """
        Substitute all resonance subgraphs to
        generate a new molecule graph

        Parameters
        ----------
        resonance_forms: List[nx.Graph]
            All disjoint ``networkx`` graphs to substitute

        Returns
        -------
        new_graph: nx.Graph
            The new molecule graph with all resonance subgraphs
        """

        atoms = {}
        bonds = {}
        for subgraph in resonance_forms:
            for node in subgraph.nodes:
                atoms[node] = subgraph.nodes[node]
            for i, j in subgraph.edges:
                key = tuple(sorted((i, j)))
                bonds[key] = subgraph.edges[i, j]
        for i, atom in self._graph_dict["atoms"].items():
            if i not in atoms:
                atoms[i] = atom
        for key, bond in self._graph_dict["bonds"].items():
            if key not in bonds:
                bonds[key] = bond
        return {"atoms": atoms, "bonds": bonds}
        
    @staticmethod
    def _update_graph_attributes(source: nx.Graph, target: nx.Graph):
        """
        Update the attributes of the nodes and edges of
        a target graph with those of a source graph.
        All nodes and edges in the source graph should be present
        in the target.
        The target graph is updated in-place.

        Parameters
        ----------
        source: :class:`networkx.Graph`
            Source sub-graph
        target: :class:`networkx.Graph`
            Target graph

        """
        for node in source.nodes:
            # target.nodes[node].update(source.nodes[node])
            target.atoms[node].update(source.nodes[node])
        for i, j in source.edges:
            key = tuple(sorted((i, j)))
            # target.edges[i, j].update(source.edges[i, j])
            target.bonds[key].update(source.edges[i, j])

    @staticmethod
    def _convert_molecule_to_graph(molecule):
        """
        Convert a molecule to a ``networkx`` graph
        where each node has ``bond_orders`` information

        Parameters
        ----------
        molecule: :class:`openff.toolkit.topology.Molecule`

        Returns
        -------
        graph: :class:`networkx.Graph`
        """
        graph = molecule.to_networkx()
        for node, atom in zip(graph.nodes, molecule.atoms):
            bond_orders = tuple(sorted(bond.bond_order for bond in atom.bonds))
            graph.nodes[node]["bond_orders"] = tuple(bond_orders)

        return graph

    @staticmethod
    def _reduce_graph(graph: nx.Graph, inplace: bool = True) -> nx.Graph:
        """
        Reduce a ``networkx`` graph by removing all hydrogen
        and CX4 atoms

        Parameters
        ----------
        graph: :class:`networkx.Graph`
            Input molecule graph with all atoms
        inplace: bool
            Whether to reduce the graph inplace

        Returns
        -------
        reduced_graph: :class:`networkx.Graph`
        """
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
        """
        Fragment a ``networkx`` graph into disjoint subgraphs
        """
        for fragment in nx.connected_components(graph):
            yield graph.subgraph(fragment)

    def _get_acceptor_donor_fragments(self) -> List["FragmentEnumerator"]:
        """
        Get all fragments of the molecule that contain both
        electron donors and acceptors

        Returns
        -------
        fragments: List["FragmentEnumerator"]
        """
        acceptor_donor_fragments = []

        for nxfragment in self._fragment_networkx_graph(self.reduced_graph):
            fragment = FragmentEnumerator(nxfragment)
            if fragment.acceptor_indices and fragment.donor_indices:
                acceptor_donor_fragments.append(fragment)

        return acceptor_donor_fragments


class FragmentEnumerator:

    """
    A convenience class to enumerate resonance forms of a fragment of a molecule.

    Parameters
    ----------
    graph: nx.Graph
        A ``networkx`` Graph representation of a molecule fragment.
    """

    def __init__(self, graph):
        self.reduced_graph = graph
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
                self.reduced_graph,
                node_a,
                node_b,
                cutoff=max_path_length,
            ),
        )
        odd_paths = [path for path in all_paths if len(path) % 2 == 1]
        odd_paths = tuple(sorted(odd_paths, key=len, reverse=True))
        self._path_cache[key] = odd_paths
        self._path_cache[key[::-1]] = odd_paths
        return odd_paths

    def _get_transfer_paths(
        self,
        donor_node: int,
        acceptor_node: int,
        max_path_length: Optional[int] = None,
    ) -> Generator[Tuple[int, ...], None, None]:
        """
        Attempts to find all possible electron transfer paths, as defined by Gilson et
        al [1], between a donor and an acceptor atom.

        Parameters
        ----------
        donor_node: int
            The node of the donor atom
        acceptor_node: int
            The node of the acceptor atom
        max_path_length: Optional[int]
            The maximum length of the paths to search.
            If None, all paths will be returned.


        Returns
        -------
        transfer_paths: Generator[Tuple[int, ...], None, None
            A list of any 'electron transfer' paths that begin from the donor atom and end
            at the acceptor atom.
        """
        for path in self._get_all_odd_n_simple_paths(
            donor_node, acceptor_node, max_path_length
        ):
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
        bond_orders = np.array(
            [self.reduced_graph.edges[i, j]["bond_order"] for i, j in edges]
        )
        deltas = bond_orders[1:] - bond_orders[:-1]
        return np.all(deltas[::2] == 1) and np.all(deltas[1::2] == -1)

    def _to_resonance_dict(
        self, include_bonds: bool = True, include_formal_charges: bool = False
    ) -> Dict[str, List[int]]:
        """
        A convenience method to convert a molecule fragment to a dictionary.
        This is used to generate a hash of the fragment.

        Parameters
        ----------
        include_bonds: bool
            Whether to include bond orders
        include_formal_charges: bool
            Whether to include atom formal charges

        Returns
        -------
        dictionary: Dict[str, List[int]]
            A dictionary representation of the fragment.
            The keys are "acceptor_indices" and "donor_indices".
            Optionally, they can include "bond_orders" and "formal_charges".
        """
        hash_dict = {
            "acceptor_indices": self.acceptor_indices,
            "donor_indices": self.donor_indices,
        }
        if include_bonds:
            bond_data = self.reduced_graph.edges(data="bond_order")
            hash_dict["bond_orders"] = {(i, j): x for i, j, x in bond_data}
        if include_formal_charges:
            hash_dict["formal_charges"] = [
                q.m_as(unit.elementary_charge)
                for _, q in self.reduced_graph.nodes(data="formal_charge")
            ]
        return hash_dict

    def _to_resonance_json(
        self, include_bonds: bool = True, include_formal_charges: bool = False
    ) -> str:
        """
        A convenience method to convert a molecule fragment to a JSON string

        Parameters
        ----------
        include_bonds: bool
            Whether to include bond orders
        include_formal_charges: bool
            Whether to include atom formal charges

        Returns
        -------
        string: str
        """
        resonance_dict = self._to_resonance_dict(
            include_bonds=include_bonds,
            include_formal_charges=include_formal_charges,
        )
        if include_bonds:
            resonance_dict["bond_orders"] = sorted(
                map(tuple, resonance_dict["bond_orders"].items())
            )
        return json.dumps(resonance_dict, sort_keys=True)

    def _to_resonance_hash(
        self, include_bonds: bool = True, include_formal_charges: bool = False
    ) -> bytes:
        """
        A convenience method to convert a molecule fragment to a hash

        Parameters
        ----------
        include_bonds: bool
            Whether to include bond orders
        include_formal_charges: bool
            Whether to include atom formal charges

        Returns
        -------
        hashed_string: bytes
        """
        import hashlib

        json_str = self._to_resonance_json(
            include_bonds=include_bonds, include_formal_charges=include_formal_charges
        )
        return hashlib.sha1(json_str.encode(), usedforsecurity=False).digest()

    def _transfer_electrons(self, path: Tuple[int, ...]):
        """
        Carries out an electron transfer along the pre-determined transfer path starting
        from a donor and ending in an acceptor.

        Parameters
        ----------
        path: Tuple[int, ...]
            The path along which to transfer electrons.
            The first and last atoms are the donor and acceptor, respectively.


        Returns
        -------
        new_fragment: FragmentEnumerator
            The new fragment after the electron transfer
        """
        graph = copy.deepcopy(self.reduced_graph)
        if not len(path):
            return type(self)(graph)

        donor_index, acceptor_index = path[0], path[-1]
        for index in [donor_index, acceptor_index]:
            resonance_type = self._get_atom_resonance_type(index)
            conjugate_key = resonance_type.get_conjugate_key()
            charge = conjugate_key.formal_charge * unit.elementary_charge
            graph.nodes[index]["formal_charge"] = charge
            graph.nodes[index]["atomic_number"] = conjugate_key.atomic_number
            graph.nodes[index]["bond_orders"] = conjugate_key.bond_orders

        for bond_index, (i, j) in enumerate(zip(path[:-1], path[1:])):
            increment = 1 if bond_index % 2 == 0 else -1
            graph.edges[i, j]["bond_order"] += increment

        return type(self)(graph)

    def _get_atom_resonance_type(self, index: int) -> ResonanceType.Value:
        """
        Get the resonance type of an atom.

        Parameters
        ----------
        node: int
            Node of the atom to get resonance type of

        Returns
        -------
        resonance_type: ResonanceType.Value
        """
        node_info = self.reduced_graph.nodes[index]
        charge = node_info["formal_charge"].m_as(unit.elementary_charge)
        return ResonanceType.get_resonance_type(
            atomic_number=node_info["atomic_number"],
            formal_charge=charge,
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
        for index in self.reduced_graph.nodes:
            try:
                resonance_types[index] = self._get_atom_resonance_type(index)
            except KeyError:
                pass
        return resonance_types

    def _enumerate_donor_acceptor_resonance_forms(
        self,
        max_path_length: Optional[int] = None,
    ) -> Generator["FragmentEnumerator", None, None]:
        """Enumerate all resonance forms of this fragment
        by transferring electrons

        Parameters
        ----------
        max_path_length: int, optional
            The maximum length of possible transfer paths.
            If None, all paths are considered.

        Returns
        -------
        resonance_forms: Generator[FragmentEnumerator, None, None]
        """
        for acceptor_index in self.acceptor_indices:
            for donor_index in self.donor_indices:
                for path in self._get_transfer_paths(
                    donor_index, acceptor_index, max_path_length
                ):
                    transferred = self._transfer_electrons(path)
                    yield transferred

    def enumerate_resonance_forms(
        self,
        include_all_transfer_pathways: bool = False,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
    ) -> List["FragmentEnumerator"]:
        """
        Recursively enumerate all resonance forms for a fragment
        using the v-charge algorithm.

        Parameters
        ----------
        include_all_transfer_pathways: bool
            Whether to include all possible transfer pathways.
            If False, drop resonance forms that only differ due to bond order
            rearrangements, e.g. ring resonance structures
        lowest_energy_only: bool
            Whether to only include the lowest energy resonance forms.
        max_path_length: int, optional
            The maximum length of possible transfer paths.
            If None, all paths are considered.

        Returns
        -------
        resonance_forms: List[FragmentEnumerator]
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

                new_forms = current_fragment._enumerate_donor_acceptor_resonance_forms(
                    max_path_length=max_path_length
                )
                for new_fragment in new_forms:
                    new_key = new_fragment._to_resonance_hash(include_bonds=True)
                    if new_key not in closed_forms and new_key not in open_forms:
                        current_forms[new_key] = new_fragment

            open_forms = current_forms

        if not include_all_transfer_pathways:
            # Drop resonance forms that only differ due to bond order re-arrangements as we
            # aren't interested in e.g. ring resonance structures.
            closed_forms = {
                fragment._to_resonance_hash(include_bonds=False): fragment
                for fragment in closed_forms.values()
            }

        if lowest_energy_only:
            closed_forms = self._select_lowest_energy_forms(closed_forms)

        graphs = list(closed_forms.values())
        return graphs

    @staticmethod
    def _select_lowest_energy_forms(
        forms: Dict[Any, "FragmentEnumerator"]
    ) -> Dict[Any, "FragmentEnumerator"]:
        """
        Select the resonance forms with the lowest energy.

        Parameters
        ----------
        forms: Dict[Any, FragmentEnumerator]
            All possible resonance forms

        Returns
        -------
        lowest_energy_forms: Dict[Any, FragmentEnumerator]
            The resonance forms with the lowest energy.
        """

        energies: Dict[Any, float] = {
            key: form._get_resonance_energy() for key, form in forms.items()
        }
        lowest = min(energies.values())
        lowest_forms = {
            key: forms[key]
            for key, energy in energies.items()
            if np.isclose(energy, lowest)
        }
        return lowest_forms

    def _get_resonance_energy(self) -> float:
        """
        Calculate the resonance energy sum of all resonance atoms in this form

        Returns
        -------
        energy: float
        """
        return sum(res.energy for res in self.resonance_types.values())
