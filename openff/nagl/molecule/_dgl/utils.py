from typing import Dict, List, TYPE_CHECKING, Optional

import torch
from openff.utilities import requires_package
from openff.toolkit.topology.molecule import Molecule

from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.features._featurizers import AtomFeaturizer, BondFeaturizer
from openff.nagl.molecule._utils import FORWARD, REVERSE, FEATURE


if TYPE_CHECKING:
    import dgl


@requires_package("dgl")
def openff_molecule_to_base_dgl_graph(
    molecule: Molecule,
    forward: str = FORWARD,
    reverse: str = REVERSE,
) -> "dgl.DGLHeteroGraph":
    """
    Convert an OpenFF Molecule to a DGL graph.
    """
    import dgl
    from openff.nagl.toolkits.openff import get_openff_molecule_bond_indices

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
    molecule: Molecule,
    atom_features: List[AtomFeature] = tuple(),
    bond_features: List[BondFeature] = tuple(),
    atom_feature_tensor: Optional[torch.Tensor] = None,
    bond_feature_tensor: Optional[torch.Tensor] = None,
    forward: str = FORWARD,
    reverse: str = REVERSE,
) -> "dgl.DGLHeteroGraph":
    from openff.nagl.molecule._utils import _get_openff_molecule_information

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

    # create base undirected graph
    molecule_graph = openff_molecule_to_base_dgl_graph(
        molecule,
        forward=forward,
        reverse=reverse,
    )

    # add atom features
    if len(atom_features):
        atom_featurizer = AtomFeaturizer(atom_features)
        atom_feature_tensor = atom_featurizer.featurize(molecule)
    molecule_graph.ndata[FEATURE]

    # add additional information
    molecule_info = _get_openff_molecule_information(molecule)
    for key, value in molecule_info.items():
        molecule_graph.ndata[key] = value

    # add bond features
    bond_orders = torch.tensor(
        [bond.bond_order for bond in molecule.bonds], dtype=torch.uint8
    )

    if len(bond_features):
        bond_featurizer = BondFeaturizer(bond_features)
        bond_feature_tensor = bond_featurizer.featurize(molecule)

    for direction in (forward, reverse):
        if bond_feature_tensor is not None:
            molecule_graph.edges[direction].data[FEATURE] = bond_feature_tensor
        molecule_graph.edges[direction].data["bond_order"] = bond_orders

    return molecule_graph


@requires_package("dgl")
def dgl_heterograph_to_homograph(graph: "dgl.DGLHeteroGraph") -> "dgl.DGLGraph":
    import dgl

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
                homo_graph = dgl.to_homogeneous(graph, ndata=[], edata=[FEATURE])
            except KeyError:
                homo_graph = dgl.to_homogeneous(graph, ndata=[], edata=[])

    return homo_graph
