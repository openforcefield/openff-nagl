import numpy as np
import pytest
import torch

from openff.nagl.dgl.molecule import DGLMolecule
from openff.nagl.nn.modules.pooling import (
    PoolAtomFeatures,
    PoolBondFeatures,
)


class BaseTestFeatures:
    @pytest.fixture
    def array(self):
        return np.arange(5).reshape(-1, 1)

    @pytest.fixture
    def featurized_molecule(self, dgl_methane, array):
        tensor = torch.from_numpy(array)
        dgl_methane.graph.ndata[DGLMolecule._graph_feature_name] = tensor
        return dgl_methane


class TestPoolAtomFeatures(BaseTestFeatures):
    def test_forward(self, featurized_molecule, array):
        features = PoolAtomFeatures().forward(featurized_molecule).numpy()
        assert np.allclose(features, array)


class TestPoolBondFeatures(BaseTestFeatures):
    def test_forward(self, featurized_molecule, array):
        bond_pool_layer = PoolBondFeatures(torch.nn.Identity(8))
        features = bond_pool_layer.forward(featurized_molecule).numpy()
        assert not np.allclose(features, 0.0)
        assert np.allclose(features[:, 0], np.arange(1, 5))
        assert np.allclose(features[:, 0], features[:, 1])
