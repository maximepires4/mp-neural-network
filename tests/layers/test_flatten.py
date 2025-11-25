import numpy as np

from mpneuralnetwork.layers import Flatten


def test_flatten_shapes():
    """
    Teste la couche Flatten (Forward et Backward).
    """
    layer = Flatten()
    input_shape = (2, 5, 5)  # (C, H, W)
    layer.build(input_shape)

    assert layer.output_size == 50

    # Forward
    batch_size = 10
    X = np.random.randn(batch_size, *input_shape)
    out = layer.forward(X)

    assert out.shape == (batch_size, 50)

    # Backward
    dout = np.random.randn(batch_size, 50)
    dx = layer.backward(dout)

    assert dx.shape == (batch_size, 2, 5, 5)
    assert np.allclose(dx.flatten(), dout.flatten())
