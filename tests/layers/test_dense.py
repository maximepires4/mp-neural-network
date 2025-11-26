import numpy as np
import pytest

from mpneuralnetwork import DTYPE
from mpneuralnetwork.layers import Dense
from mpneuralnetwork.losses import MSE

np.random.seed(69)


def test_dense_config():
    """Test get_config for Dense layer."""
    dense = Dense(10, input_size=5, initialization="he")
    config = dense.get_config()
    assert config["output_size"] == 10
    assert config["input_size"] == 5
    assert config["initialization"] == "he"
    assert config["type"] == "Dense"


@pytest.mark.parametrize(
    "input_data, weights, biases, expected_shape, expected_output",
    [
        (np.random.randn(64, 10).astype(DTYPE), np.random.randn(10, 5).astype(DTYPE), np.random.randn(1, 5).astype(DTYPE), (64, 5), None),
        (
            np.array([[0.2, 0.8]], dtype=DTYPE),
            np.array([[0.5], [0.5]], dtype=DTYPE),
            np.array([[0.1]], dtype=DTYPE),
            (1, 1),
            np.array([[0.6]], dtype=DTYPE),
        ),
    ],
)
def test_dense_forward_pass(input_data, weights, biases, expected_shape, expected_output):
    n_inputs = input_data.shape[1]
    n_outputs = expected_shape[1]
    layer = Dense(n_outputs, input_size=n_inputs, initialization="xavier")
    layer.weights = weights
    layer.biases = biases

    output = layer.forward(input_data)

    assert output.shape == expected_shape, "Output shape is incorrect"
    if expected_output is not None:
        assert np.allclose(output, expected_output), "Forward pass calculation is incorrect"


def test_dense_gradient_checking():
    """
    Performs numerical gradient checking for the Dense layer.
    """
    batch_size, n_inputs, n_outputs = 4, 5, 3
    layer = Dense(n_outputs, input_size=n_inputs, initialization="xavier")
    loss_fn = MSE()
    epsilon = 1e-3

    X = np.random.randn(batch_size, n_inputs).astype(DTYPE)
    Y = np.random.randn(batch_size, n_outputs).astype(DTYPE)

    # --- Check Weights Gradient (d_loss / d_w) ---
    numerical_grads_w = np.zeros_like(layer.weights, dtype=DTYPE)
    for i in range(layer.weights.shape[0]):
        for j in range(layer.weights.shape[1]):
            original_w = layer.weights[i, j]

            layer.weights[i, j] = original_w + epsilon
            loss_plus = loss_fn.direct(layer.forward(X), Y)

            layer.weights[i, j] = original_w - epsilon
            loss_minus = loss_fn.direct(layer.forward(X), Y)

            layer.weights[i, j] = original_w
            numerical_grads_w[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    preds = layer.forward(X)
    output_gradient = loss_fn.prime(preds, Y)
    layer.backward(output_gradient)
    analytical_grads_w = layer.weights_gradient

    assert np.allclose(analytical_grads_w, numerical_grads_w, atol=1e-2), "Weight gradients do not match"
    # --- Check Biases Gradient (d_loss / d_b) ---
    numerical_grads_b = np.zeros_like(layer.biases, dtype=DTYPE)
    for i in range(layer.biases.shape[1]):
        original_b = layer.biases[0, i]

        layer.biases[0, i] = original_b + epsilon
        loss_plus = loss_fn.direct(layer.forward(X), Y)

        layer.biases[0, i] = original_b - epsilon
        loss_minus = loss_fn.direct(layer.forward(X), Y)

        layer.biases[0, i] = original_b
        numerical_grads_b[0, i] = (loss_plus - loss_minus) / (2 * epsilon)

    analytical_grads_b = layer.biases_gradient
    assert np.allclose(analytical_grads_b, numerical_grads_b, atol=1e-2), "Bias gradients do not match"

    # --- Check Input Gradient (d_loss / d_x) ---
    numerical_grads_x = np.zeros_like(X, dtype=DTYPE)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            original_x = X[i, j]

            X[i, j] = original_x + epsilon
            loss_plus = loss_fn.direct(layer.forward(X), Y)

            X[i, j] = original_x - epsilon
            loss_minus = loss_fn.direct(layer.forward(X), Y)

            X[i, j] = original_x
            numerical_grads_x[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    analytical_grads_x = layer.backward(output_gradient)
    assert np.allclose(analytical_grads_x, numerical_grads_x, atol=1e-2), "Input gradients do not match"
