import numpy as np

from mpneuralnetwork import DTYPE
from mpneuralnetwork.layers import Dropout
from mpneuralnetwork.losses import MSE

np.random.seed(69)


def test_dropout_config_and_inference():
    """Test get_config and inference behavior for Dropout."""
    dropout = Dropout(probability=0.3)
    config = dropout.get_config()
    assert config["probability"] == 0.3
    assert config["type"] == "Dropout"

    # Test inference
    dropout_inf = Dropout(0.5)
    x = np.ones((5, 5))
    # During inference (training=False), output should equal input
    out = dropout_inf.forward(x, training=False)
    assert np.array_equal(out, x)


def test_dropout_gradient():
    """
    Performs numerical gradient checking for the Dropout layer.
    """
    batch_size, n_features = 4, 10
    layer = Dropout(probability=0.5)
    loss_fn = MSE()

    X = np.random.randn(batch_size, n_features).astype(DTYPE)
    Y = np.random.randn(batch_size, n_features).astype(DTYPE)

    # --- Analytical Gradient ---
    preds = layer.forward(X.copy(), training=True)
    output_gradient = loss_fn.prime(preds, Y)
    analytical_grads = layer.backward(output_gradient)

    # --- Numerical Gradient ---
    fixed_mask = layer.mask
    epsilon = 1e-3
    numerical_grads = np.zeros_like(X, dtype=DTYPE)

    it = np.nditer(X, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        original_value = X[ix]

        X[ix] = original_value + epsilon
        preds_plus = X * fixed_mask
        loss_plus = np.sum(loss_fn.direct(preds_plus, Y))

        X[ix] = original_value - epsilon
        preds_minus = X * fixed_mask
        loss_minus = np.sum(loss_fn.direct(preds_minus, Y))

        X[ix] = original_value
        numerical_grads[ix] = (loss_plus - loss_minus) / (2 * epsilon)
        it.iternext()

    assert np.allclose(analytical_grads, numerical_grads, atol=1e-2), "Dropout gradient does not match"
