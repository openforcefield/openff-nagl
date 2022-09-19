import pytest
from openff.toolkit.topology import Molecule as OFFMolecule

from gnn_charge_models.resonance.paths import PathGenerator


class TestPathGenerator:
    @pytest.fixture()
    def path_generator(self):
        offmol = OFFMolecule.from_mapped_smiles("[C:1]([O-:2])(=[O:3])([H:4])")
        return PathGenerator(offmol.to_rdkit())

    @pytest.mark.parametrize(
        "source, target, cutoff, expected_paths",
        [(1, 2, None, ((1, 0, 2),)), (2, 1, None, ((2, 0, 1),)), (1, 2, 1, tuple())],
    )
    def test_all_odd_node_simple_paths(
        self, path_generator, source, target, cutoff, expected_paths
    ):
        paths = path_generator.all_odd_node_simple_paths(source, target, cutoff)
        assert paths == expected_paths
