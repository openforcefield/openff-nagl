from openff.nagl.molecule._graph.molecule import (
    GraphMolecule,
    GraphMoleculeBatch,
)


class TestNXMoleculeBatch:
    def test_from_molecules(self):
        mol1 = GraphMolecule.from_smiles("C")
        mol2 = GraphMolecule.from_smiles("CC")

        batch = GraphMoleculeBatch.from_nx_molecules([mol1, mol2])
        assert batch.graph.batch_size == 2
        assert batch.n_atoms == (5, 8)
        assert batch.n_representations == (1, 1)
