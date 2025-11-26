import numpy as np

from mpneuralnetwork.layers import BatchNormalization
from mpneuralnetwork.losses import MSE
from tests.utils import check_gradient

np.random.seed(69)


def test_bn_config():
    """Test get_config for BatchNormalization layer."""
    bn = BatchNormalization(momentum=0.8, epsilon=1e-4)
    config = bn.get_config()
    assert config["momentum"] == 0.8
    assert config["epsilon"] == 1e-4
    assert config["type"] == "BatchNormalization"


def test_bn_forward_properties():
    """Test that BN output has zero mean and unit variance during training."""
    bn = BatchNormalization()
    bn.build(10)

    X = np.random.randn(100, 10) * 5 + 3
    out = bn.forward(X, training=True)

    assert np.allclose(np.mean(out, axis=0), 0, atol=0.1)
    assert np.allclose(np.std(out, axis=0), 1, atol=0.1)

    assert np.all(np.abs(bn.cache_m) > 0)
    assert np.all(bn.cache_v != 1.0)


def test_bn_gradient_check():
    """
    Performs numerical gradient checking for BatchNormalization.
    """
    batch_size, n_features = 4, 3
    layer = BatchNormalization()
    layer.build(n_features)
    loss_fn = MSE()

    X = np.random.randn(batch_size, n_features)
    Y = np.random.randn(batch_size, n_features)

    layer.gamma = np.random.randn(1, n_features)
    layer.beta = np.random.randn(1, n_features)

    check_gradient(layer, X, Y, loss_fn)
