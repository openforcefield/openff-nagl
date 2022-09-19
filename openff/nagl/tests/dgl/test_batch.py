import dgl
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from openff.nagl.dgl.batch import DGLMolecule, DGLMoleculeBatch
from openff.nagl.features import AtomConnectivity, BondIsInRing


class TestDGLMoleculeBatch:
    def test_from_molecules(self):

        mol1 = DGLMolecule.from_smiles("C")
        mol2 = DGLMolecule.from_smiles("CC")

        batch = DGLMoleculeBatch.from_dgl_molecules([mol1, mol2])
        assert batch.graph.batch_size == 2
        assert batch.n_atoms == (5, 8)
        assert batch.n_representations == (1, 1)
