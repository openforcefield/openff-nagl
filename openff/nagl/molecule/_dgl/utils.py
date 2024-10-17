from typing import Dict, List, TYPE_CHECKING, Optional

import torch
import numpy as np
from openff.utilities import requires_package

from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.features._featurizers import AtomFeaturizer, BondFeaturizer
from openff.nagl.molecule._utils import FORWARD, REVERSE, FEATURE


if TYPE_CHECKING:
    import dgl
    from openff.toolkit.topology.molecule import Molecule



@requires_package("dgl")
def openff_molecule_to_base_dgl_graph(
    molecule: "Molecule",
    forward: str = FORWARD,
    reverse: str = REVERSE,
) -> "dgl.DGLHeteroGraph":
    """
    Convert an OpenFF Molecule to a DGL graph.
    """
    import dgl
    from openff.nagl.toolkits.openff import get_openff_molecule_bond_indices

    bonds = get_openff_molecule_bond_indices(molecule)
    if bonds:
        indices_a, indices_b = map(list, zip(*bonds))
    else:
        indices_a, indices_b = [], []
    indices_a = torch.tensor(indices_a, dtype=torch.int32)
    indices_b = torch.tensor(indices_b, dtype=torch.int32)

    molecule_graph = dgl.heterograph(
        {
            ("atom", forward, "atom"): (indices_a, indices_b),
            ("atom", reverse, "atom"): (indices_b, indices_a),
        },
        num_nodes_dict={"atom": molecule.n_atoms},
    )
    return molecule_graph


def openff_molecule_to_dgl_graph(
    molecule: "Molecule",
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
    
    if atom_feature_tensor is None:
        atom_feature_tensor = torch.zeros((molecule.n_atoms, 0))
    molecule_graph.ndata[FEATURE] = atom_feature_tensor.reshape(molecule.n_atoms, -1)

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
        n_bonds = len(molecule.bonds)
        if bond_feature_tensor is not None and n_bonds:
            bond_feature_tensor = bond_feature_tensor.reshape(n_bonds, -1)
        else:
            bond_feature_tensor = torch.zeros((n_bonds, 0))
        molecule_graph.edges[direction].data[FEATURE] = bond_feature_tensor
        molecule_graph.edges[direction].data["bond_order"] = bond_orders

    return molecule_graph

@requires_package("dgl")
def heterograph_to_homograph_no_edges(G: "dgl.DGLHeteroGraph", ndata=None, edata=None) -> "dgl.DGLGraph":
    """
    Copied and modified from dgl.python.dgl.convert.to_homogeneous,
    but with the edges removed.

    This part of the code is licensed under the Apache 2.0 license according
    to the terms of DGL (https://github.com/dmlc/dgl?tab=Apache-2.0-1-ov-file).

    Please see our third-party license file for more information
    (https://github.com/openforcefield/openff-nagl/blob/main/LICENSE-3RD-PARTY)
    """
    import dgl
    from dgl import backend as F
    from dgl.base import EID, NID, ETYPE, NTYPE
    from dgl.heterograph import combine_frames

    # TODO: revisit in case DGL accounts for this in the future
    num_nodes_per_ntype = [G.num_nodes(ntype) for ntype in G.ntypes]
    offset_per_ntype = np.insert(np.cumsum(num_nodes_per_ntype), 0, 0)
    srcs = []
    dsts = []
    nids = []
    eids = []
    ntype_ids = []
    etype_ids = []
    total_num_nodes = 0

    for ntype_id, ntype in enumerate(G.ntypes):
        num_nodes = G.num_nodes(ntype)
        total_num_nodes += num_nodes
        # Type ID is always in int64
        ntype_ids.append(F.full_1d(num_nodes, ntype_id, F.int64, G.device))
        nids.append(F.arange(0, num_nodes, G.idtype, G.device))

    for etype_id, etype in enumerate(G.canonical_etypes):
        srctype, _, dsttype = etype
        src, dst = G.all_edges(etype=etype, order="eid")
        num_edges = len(src)
        srcs.append(src + int(offset_per_ntype[G.get_ntype_id(srctype)]))
        dsts.append(dst + int(offset_per_ntype[G.get_ntype_id(dsttype)]))
        etype_ids.append(F.full_1d(num_edges, etype_id, F.int64, G.device))
        eids.append(F.arange(0, num_edges, G.idtype, G.device))

    retg = dgl.graph(
        (F.cat(srcs, 0), F.cat(dsts, 0)),
        num_nodes=total_num_nodes,
        idtype=G.idtype,
        device=G.device,
    )

    # copy features
    if ndata is None:
        ndata = []
    if edata is None:
        edata = []
    comb_nf = combine_frames(
        G._node_frames, range(len(G.ntypes)), col_names=ndata
    )
    if comb_nf is not None:
        retg.ndata.update(comb_nf)

    retg.ndata[NID] = F.cat(nids, 0)
    retg.edata[EID] = F.cat(eids, 0)
    retg.ndata[NTYPE] = F.cat(ntype_ids, 0)
    retg.edata[ETYPE] = F.cat(etype_ids, 0)

    return retg




@requires_package("dgl")
def dgl_heterograph_to_homograph(graph: "dgl.DGLHeteroGraph") -> "dgl.DGLGraph":
    import dgl

    try:
        homo_graph = dgl.to_homogeneous(graph, ndata=[FEATURE], edata=[FEATURE])
    except TypeError as e:
        if graph.num_edges() == 0:
            homo_graph = heterograph_to_homograph_no_edges(graph)
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
