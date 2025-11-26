import numpy as np

from mpneuralnetwork.layers.layer2d import BatchNormalization2D


def test_bn2d_config():
    layer = BatchNormalization2D(momentum=0.8, epsilon=1e-4)
    config = layer.get_config()
    assert config["momentum"] == 0.8
    assert config["epsilon"] == 1e-4


def test_bn2d_shape():
    N, C, H, W = 5, 3, 10, 10
    layer = BatchNormalization2D()
    layer.build((C, H, W))

    X = np.random.randn(N, C, H, W)
    out = layer.forward(X)

    assert out.shape == X.shape

    dout = np.random.randn(*out.shape)
    dx = layer.backward(dout)

    assert dx.shape == X.shape
    assert layer.gamma_gradient.shape == layer.gamma.shape
    assert layer.beta_gradient.shape == layer.beta.shape
    assert layer.gamma.shape == (1, C, 1, 1)


def test_bn2d_normalization():
    """Test that normalization happens per channel over (N, H, W)"""
    N, C, H, W = 100, 2, 5, 5
    layer = BatchNormalization2D()
    layer.build((C, H, W))

    # Create data with specific mean/std per channel
    # Channel 0: Mean 10, Std 2
    # Channel 1: Mean -5, Std 5
    X = np.zeros((N, C, H, W))
    X[:, 0, :, :] = np.random.randn(N, H, W) * 2 + 10
    X[:, 1, :, :] = np.random.randn(N, H, W) * 5 - 5

    out = layer.forward(X, training=True)

    # Check outputs (should be mean 0, std 1 per channel)
    # Tolerance higher due to batch estimation
    assert np.allclose(np.mean(out[:, 0, :, :]), 0, atol=0.2)
    assert np.allclose(np.std(out[:, 0, :, :]), 1, atol=0.2)

    assert np.allclose(np.mean(out[:, 1, :, :]), 0, atol=0.2)
    assert np.allclose(np.std(out[:, 1, :, :]), 1, atol=0.2)


def test_bn2d_running_stats():
    layer = BatchNormalization2D(momentum=0.9)
    layer.build((2, 5, 5))

    X = np.random.randn(10, 2, 5, 5)
    layer.forward(X, training=True)

    # Check that cache is updated
    assert np.any(layer.cache_m != 0)
    assert np.any(layer.cache_v != 1)

    # Check inference uses cache
    # We manually set cache to something obvious
    layer.cache_m[:] = 100.0
    layer.cache_v[:] = 1.0

    # Input 100 should become 0 if using cache
    X_test = np.ones((1, 2, 5, 5)) * 100.0
    out = layer.forward(X_test, training=False)

    assert np.allclose(out, 0.0)
