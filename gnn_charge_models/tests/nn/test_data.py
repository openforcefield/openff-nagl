import os

import pytest
import torch
import numpy as np

from gnn_charge_models.nn.data import DGLMoleculeDataset
from gnn_charge_models.dgl import DGLMolecule
from gnn_charge_models.features import AtomConnectivity, BondIsInRing
from gnn_charge_models.storage.store import MoleculeStore, MoleculeRecord, ConformerRecord, PartialChargeRecord, WibergBondOrderRecord


from openff.toolkit.topology.molecule import Molecule as OFFMolecule, unit as off_unit


def label_formal_charge(molecule: OFFMolecule):
    return {
        "formal_charges": torch.tensor(
            [
                atom.formal_charge.value_in_unit(off_unit.elementary_charge)
                for atom in molecule.atoms
            ],
            dtype=torch.float,
        ),
    }


def test_data_set_from_molecules(openff_methane_charged):

    data_set = DGLMoleculeDataset.from_openff(
        [openff_methane_charged],
        label_function=label_formal_charge,
        atom_features=[AtomConnectivity()],
        bond_features=[BondIsInRing()],
    )
    assert len(data_set) == 1
    assert data_set.n_features == 4

    dgl_molecule, labels = data_set[0]
    assert isinstance(dgl_molecule, DGLMolecule)
    assert dgl_molecule.n_atoms == 5

    assert "formal_charges" in labels
    label = labels["formal_charges"]
    assert label.numpy().shape == (5,)


def test_data_set_from_molecule_stores(tmpdir):

    charges = PartialChargeRecord(method="am1", values=[0.1, -0.1])
    bond_orders = WibergBondOrderRecord(
        method="am1",
        values=[(0, 1, 1.1)],
    )
    molecule_record = MoleculeRecord(
        mapped_smiles="[Cl:1]-[H:2]",
        conformers=[
            ConformerRecord(
                coordinates=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                partial_charges=[charges],
                bond_orders=[bond_orders],
            )
        ],
    )

    molecule_store = MoleculeStore(os.path.join(tmpdir, "store.sqlite"))
    molecule_store.store(records=[molecule_record])

    data_set = DGLMoleculeDataset.from_molecule_stores(
        molecule_stores=[molecule_store],
        atom_features=[AtomConnectivity()],
        bond_features=[BondIsInRing()],
        partial_charge_method="am1",
        bond_order_method="am1",

    )

    assert len(data_set) == 1
    assert data_set.n_features == 4

    dgl_molecule, labels = data_set[0]

    assert isinstance(dgl_molecule, DGLMolecule)
    assert dgl_molecule.n_atoms == 2
    assert "am1-charges" in labels
    assert labels["am1-charges"].numpy().shape == (2,)
    assert np.allclose(labels["am1-charges"].numpy(), [0.1, -0.1])
    assert "am1-wbo" in labels
    assert labels["am1-wbo"].numpy().shape == (1,)
    assert np.allclose(labels["am1-wbo"].numpy(), [1.1])
