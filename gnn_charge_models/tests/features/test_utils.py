from numpy.testing import assert_allclose


from gnn_charge_models.features.utils import one_hot_encode


def test_one_hot_encode():
    encoding = one_hot_encode("b", ["a", "b", "c"]).numpy()
    assert_allclose(encoding, [0, 1, 0])
