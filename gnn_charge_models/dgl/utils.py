from typing import List

import dgl.function
import torch

from openff.toolkit.topology.molecule import (
    Molecule as OFFMolecule,
)

from ..utils.openff import (
    get_openff_molecule_bond_indices,
    get_openff_molecule_information,
)
from ..features.atoms import AtomFeature
from ..features.bonds import BondFeature
from ..features.featurizers import AtomFeaturizer, BondFeaturizer

FORWARD = "forward"
REVERSE = "reverse"
FEATURE = "feat"


def openff_molecule_to_base_dgl_graph(
    molecule: OFFMolecule,
    forward: str = FORWARD,
    reverse: str = REVERSE,
) -> dgl.DGLHeteroGraph:
    """
    Convert an OpenFF Molecule to a DGL graph.
    """

    bonds = get_openff_molecule_bond_indices(molecule)
    indices_a, indices_b = map(list, zip(*bonds))
    indices_a = torch.tensor(indices_a, dtype=torch.int32)
    indices_b = torch.tensor(indices_b, dtype=torch.int32)

    molecule_graph = dgl.heterograph(
        {
            ("atom", forward, "atom"): (indices_a, indices_b),
            ("atom", reverse, "atom"): (indices_b, indices_a),
        }
    )
    return molecule_graph


def openff_molecule_to_dgl_graph(
    molecule: OFFMolecule,
    atom_features: List[AtomFeature] = tuple(),
    bond_features: List[BondFeature] = tuple(),
    forward: str = FORWARD,
    reverse: str = REVERSE,
) -> dgl.DGLHeteroGraph:
    # create base undirected graph
    molecule_graph = openff_molecule_to_base_dgl_graph(molecule)

    # add atom features
    if len(atom_features):
        molecule_graph.ndata[FEATURE] = AtomFeaturizer.featurize(
            molecule, atom_features
        )

    # add additional information
    molecule_info = get_openff_molecule_information(molecule)
    for key, value in molecule_info.items():
        molecule_graph.ndata[key] = value

    # add bond features
    bond_orders = torch.tensor(
        [bond.bond_order for bond in molecule.bonds],
        dtype=torch.uint8
    )

    bond_feature_tensor = None
    if len(bond_features):
        bond_feature_tensor = BondFeaturizer.featurize(molecule, bond_features)

    for direction in (forward, reverse):
        if bond_feature_tensor is not None:
            molecule_graph.edges[direction].data[FEATURE] = bond_feature_tensor
        molecule_graph.edges[direction].data["bond_order"] = bond_orders

    return molecule_graph


def dgl_heterograph_to_homograph(graph: dgl.DGLHeteroGraph) -> dgl.DGLGraph:
    try:
        homo_graph = dgl.to_homogeneous(graph, ndata=[FEATURE], edata=[FEATURE])
    except KeyError:

        # A nasty workaround to check when we don't have any atom / bond features as
        # DGL doesn't allow easy querying of features dicts for hetereographs with
        # multiple edge / node types.
        try:
            homo_graph = dgl.to_homogeneous(graph, ndata=[FEATURE], edata=[])
        except KeyError:
            try:
                homo_graph = dgl.to_homogeneous(
                    graph, ndata=[], edata=[FEATURE])
            except KeyError:
                homo_graph = dgl.to_homogeneous(graph, ndata=[], edata=[])

    else:
        return homo_graph
