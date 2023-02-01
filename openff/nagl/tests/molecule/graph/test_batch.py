
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from openff.nagl.molecule._graph.molecule import NXMolecule, NXMoleculeBatch

class TestNXMoleculeBatch:
    def test_from_molecules(self):

        mol1 = NXMolecule.from_smiles("C")
        mol2 = NXMolecule.from_smiles("CC")

        batch = NXMoleculeBatch.from_nx_molecules([mol1, mol2])
        assert batch.graph.batch_size == 2
        assert batch.n_atoms == (5, 8)
        assert batch.n_representations == (1, 1)
