import numpy as np
from numpy.testing import assert_equal

from openff.nagl.features._featurizers import AtomFeaturizer, BondFeaturizer
from openff.nagl.features.atoms import AtomConnectivity, AtomIsAromatic
from openff.nagl.features.bonds import BondIsInRing, BondOrder


def test_atomfeaturizer(openff_methane_uncharged):
    featurizer = AtomFeaturizer(features=[AtomConnectivity(), AtomIsAromatic()])
    features = featurizer(openff_methane_uncharged).numpy()
    assert features.shape == (5, 5)
    connectivity = [
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ]
    aromaticity = [[0], [0], [0], [0], [0]]

    assert_equal(features, np.hstack([connectivity, aromaticity]))


def test_bondfeaturizer(openff_methane_uncharged):
    featurizer = BondFeaturizer(
        features=[BondOrder(categories=[0, 1, 2]), BondIsInRing()]
    )
    features = featurizer(openff_methane_uncharged).numpy()
    assert features.shape == (4, 4)

    orders = [[0, 1, 0]] * 4
    rings = [[0], [0], [0], [0]]

    assert_equal(features, np.hstack([orders, rings]))
