import numpy as np
import pytest
import torch

from openff.nagl.molecule._dgl.molecule import DGLMolecule
from openff.nagl.molecule._graph.molecule import GraphMolecule
from openff.nagl.nn._pooling import PoolAtomFeatures, PoolBondFeatures


class BaseTestFeatures:
    @pytest.fixture
    def array(self):
        return np.arange(5).reshape(-1, 1)

    @pytest.fixture
    def featurized_molecule_dgl(self, dgl_methane, array):
        tensor = torch.from_numpy(array)
        dgl_methane.graph.ndata[DGLMolecule._graph_feature_name] = tensor
        return dgl_methane

    @pytest.fixture
    def featurized_molecule_nx(self, nx_methane, array):
        tensor = torch.from_numpy(array)
        nx_methane.graph.ndata[GraphMolecule._graph_feature_name] = tensor
        return nx_methane


class TestPoolAtomFeatures(BaseTestFeatures):
    def test_forward_dgl(self, featurized_molecule_dgl, array):
        features = PoolAtomFeatures().forward(featurized_molecule_dgl).numpy()
        assert np.allclose(features, array)

    def test_forward_nagl(self, featurized_molecule_nx, array):
        features = PoolAtomFeatures().forward(featurized_molecule_nx).numpy()
        assert np.allclose(features, array)


class TestPoolBondFeatures(BaseTestFeatures):
    def test_forward_dgl(self, featurized_molecule_dgl, array):
        bond_pool_layer = PoolBondFeatures(torch.nn.Identity(8))
        features = bond_pool_layer.forward(featurized_molecule_dgl).numpy()
        assert not np.allclose(features, 0.0)
        assert np.allclose(features[:, 0], np.arange(1, 5))
        assert np.allclose(features[:, 0], features[:, 1])
