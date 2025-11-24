import numpy as np
import pytest

from mpneuralnetwork.layers import BatchNormalization, Dense, Dropout, Layer
from mpneuralnetwork.losses import MSE

np.random.seed(69)  # For reproducible test data in parametrization


def test_layer_base_methods():
    """Test get_config and build for layers."""
    layer = Layer()
    assert layer.get_config() == {"type": "Layer"}
    layer.build(10)
    assert layer.input_size == 10
    assert layer.output_size == 10
    assert layer.params == {}


def test_dense_config():
    """Test get_config for Dense layer."""
    dense = Dense(10, input_size=5, initialization="he")
    config = dense.get_config()
    assert config["output_size"] == 10
    assert config["input_size"] == 5
    assert config["initialization"] == "he"
    assert config["type"] == "Dense"


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


@pytest.mark.parametrize(
    "input_data, weights, biases, expected_shape, expected_output",
    [
        # Case 1: Shape check with a batch
        (np.random.randn(64, 10), np.random.randn(10, 5), np.random.randn(1, 5), (64, 5), None),
        # Case 2: Forward pass value check
        (
            np.array([[0.2, 0.8]]),  # input
            np.array([[0.5], [0.5]]),  # weights
            np.array([[0.1]]),  # biases
            (1, 1),  # expected_shape
            np.array([[0.6]]),  # expected_output: (0.2*0.5 + 0.8*0.5) + 0.1 = 0.6
        ),
    ],
)
def test_dense_forward_pass(input_data, weights, biases, expected_shape, expected_output):
    """
    Tests the forward pass for the Dense layer, checking both output shape and specific values.
    """
    # 1. Setup
    n_inputs = input_data.shape[1]
    n_outputs = expected_shape[1]
    layer = Dense(n_outputs, input_size=n_inputs, initialization="xavier")
    layer.weights = weights
    layer.biases = biases

    # 2. Action
    output = layer.forward(input_data)

    # 3. Assert
    assert output.shape == expected_shape, "Output shape is incorrect"
    if expected_output is not None:
        assert np.allclose(output, expected_output), "Forward pass calculation is incorrect"


def test_dense_gradient_checking():
    """
    Performs numerical gradient checking for the Dense layer's backward pass.
    This ensures the analytical gradients match the numerically computed ones.
    """
    # 1. Setup
    batch_size, n_inputs, n_outputs = 4, 5, 3
    layer = Dense(n_outputs, input_size=n_inputs, initialization="xavier")
    loss_fn = MSE()
    epsilon = 1e-5

    # 2. Create random data
    X = np.random.randn(batch_size, n_inputs)
    Y = np.random.randn(batch_size, n_outputs)

    # --- Check Weights Gradient (d_loss / d_w) ---
    # 3a. Calculate numerical gradient for weights
    numerical_grads_w = np.zeros_like(layer.weights)
    for i in range(layer.weights.shape[0]):
        for j in range(layer.weights.shape[1]):
            layer.weights[i, j] += epsilon
            loss_plus = loss_fn.direct(layer.forward(X), Y)

            layer.weights[i, j] -= 2 * epsilon
            loss_minus = loss_fn.direct(layer.forward(X), Y)

            layer.weights[i, j] += epsilon  # Restore
            numerical_grads_w[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    # 3b. Calculate analytical gradient
    preds = layer.forward(X)
    output_gradient = loss_fn.prime(preds, Y)
    layer.backward(output_gradient)
    analytical_grads_w = layer.weights_gradient

    # 3c. Assert gradients are close
    assert np.allclose(analytical_grads_w, numerical_grads_w, atol=epsilon), "Weight gradients do not match"

    # --- Check Biases Gradient (d_loss / d_b) ---
    # 4a. Calculate numerical gradient for biases
    numerical_grads_b = np.zeros_like(layer.biases)
    for i in range(layer.biases.shape[1]):
        layer.biases[0, i] += epsilon
        loss_plus = loss_fn.direct(layer.forward(X), Y)

        layer.biases[0, i] -= 2 * epsilon
        loss_minus = loss_fn.direct(layer.forward(X), Y)

        layer.biases[0, i] += epsilon  # Restore
        numerical_grads_b[0, i] = (loss_plus - loss_minus) / (2 * epsilon)

    # 4b. Get analytical gradient (already computed in the backward pass above)
    analytical_grads_b = layer.biases_gradient

    # 4c. Assert gradients are close
    assert np.allclose(analytical_grads_b, numerical_grads_b, atol=epsilon), "Bias gradients do not match"

    # --- Check Input Gradient (d_loss / d_x) ---
    # 5a. Calculate numerical gradient for inputs
    numerical_grads_x = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] += epsilon
            loss_plus = loss_fn.direct(layer.forward(X), Y)

            X[i, j] -= 2 * epsilon
            loss_minus = loss_fn.direct(layer.forward(X), Y)

            X[i, j] += epsilon  # Restore
            numerical_grads_x[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    # 5b. Get analytical gradient
    analytical_grads_x = layer.backward(output_gradient)

    # 5c. Assert gradients are close
    assert np.allclose(analytical_grads_x, numerical_grads_x, atol=epsilon), "Input gradients do not match"


def _check_gradient(layer, x, y, loss_fn, epsilon=1e-5, atol=1e-5):
    """
    Helper function to perform numerical gradient checking on the input gradient (dL/dX).
    """
    # 1. Get analytical gradient
    preds = layer.forward(x.copy())
    output_gradient = loss_fn.prime(preds, y)
    analytical_grads = layer.backward(output_gradient)

    # 2. Get numerical gradient
    numerical_grads = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        original_value = x[ix]

        x[ix] = original_value + epsilon
        preds_plus = layer.forward(x.copy())
        loss_plus = np.sum(loss_fn.direct(preds_plus, y))

        x[ix] = original_value - epsilon
        preds_minus = layer.forward(x.copy())
        loss_minus = np.sum(loss_fn.direct(preds_minus, y))

        x[ix] = original_value

        numerical_grads[ix] = (loss_plus - loss_minus) / (2 * epsilon)
        it.iternext()

    # 3. Assert that the gradients are close
    assert np.allclose(analytical_grads, numerical_grads, atol=atol), f"Gradient mismatch for layer {layer.__class__.__name__}"


def test_dropout_gradient():
    """
    Performs numerical gradient checking for the Dropout layer.
    This test is specific because it must handle the stochastic mask correctly.
    """
    np.random.seed(69)
    batch_size, n_features = 4, 10

    layer = Dropout(probability=0.5)
    loss_fn = MSE()

    X = np.random.randn(batch_size, n_features)
    Y = np.random.randn(batch_size, n_features)

    # --- Analytical Gradient ---
    # 1. Do a forward pass to generate and store the dropout mask internally.
    preds = layer.forward(X.copy(), training=True)

    # 2. Get the gradient from the loss function.
    output_gradient = loss_fn.prime(preds, Y)

    # 3. Calculate the analytical gradient using the backward method.
    analytical_grads = layer.backward(output_gradient)

    # --- Numerical Gradient ---
    # Use the exact same mask that was generated during the forward pass.
    fixed_mask = layer.mask
    epsilon = 1e-5
    numerical_grads = np.zeros_like(X)

    it = np.nditer(X, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        original_value = X[ix]

        # Calculate loss for x + epsilon, using the fixed mask
        X[ix] = original_value + epsilon
        preds_plus = X * fixed_mask
        loss_plus = np.sum(loss_fn.direct(preds_plus, Y))

        # Calculate loss for x - epsilon, using the fixed mask
        X[ix] = original_value - epsilon
        preds_minus = X * fixed_mask
        loss_minus = np.sum(loss_fn.direct(preds_minus, Y))

        # Restore original value and compute numerical gradient
        X[ix] = original_value
        numerical_grads[ix] = (loss_plus - loss_minus) / (2 * epsilon)

        it.iternext()

    # --- Assert ---
    assert np.allclose(analytical_grads, numerical_grads, atol=epsilon), "Dropout gradient does not match"


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

    # Create data with non-zero mean and non-unit variance
    X = np.random.randn(100, 10) * 5 + 3

    out = bn.forward(X, training=True)

    # Check mean and std
    assert np.allclose(np.mean(out, axis=0), 0, atol=0.1)
    assert np.allclose(np.std(out, axis=0), 1, atol=0.1)

    # Check that running stats are updated
    # Initial cache_m is 0. After one update with momentum 0.9:
    # new_cache = 0.9 * 0 + 0.1 * batch_mean (~3) = 0.3
    assert np.all(np.abs(bn.cache_m) > 0)
    assert np.all(bn.cache_v != 1.0)


def test_bn_gradient_check():
    """
    Performs numerical gradient checking for BatchNormalization.
    """
    np.random.seed(42)
    batch_size, n_features = 4, 3
    layer = BatchNormalization()
    layer.build(n_features)
    loss_fn = MSE()

    X = np.random.randn(batch_size, n_features)
    Y = np.random.randn(batch_size, n_features)

    # Initialize gamma/beta non-trivially
    layer.gamma = np.random.randn(1, n_features)
    layer.beta = np.random.randn(1, n_features)

    _check_gradient(layer, X, Y, loss_fn, atol=1e-3)


@pytest.mark.parametrize(
    "layer, input_shape, expected_output_shape",
    [
        # Dense Layer
        (Dense(5, input_size=10, initialization="xavier"), (64, 10), (64, 5)),
        (Dense(1, input_size=3, initialization="xavier"), (1, 3), (1, 1)),
        # Dropout Layer (should not change shape)
        (Dropout(0.5), (128, 20), (128, 20)),
        # BatchNormalization (should not change shape)
        (BatchNormalization(), (32, 10), (32, 10)),
    ],
)
def test_layer_output_shapes(layer, input_shape, expected_output_shape):
    """
    Tests that the output shape of a layer's forward pass is correct.
    """
    # 0. Build layer if needed
    if isinstance(layer, BatchNormalization):
        layer.build(input_shape[1])

    # 1. Arrange: Create random input data with the specified shape
    input_data = np.random.randn(*input_shape)

    # 2. Act: Perform a forward pass
    output = layer.forward(input_data)

    # 3. Assert: Check if the output shape matches the expected shape
    assert output.shape == expected_output_shape, (
        f"Shape mismatch for layer {layer.__class__.__name__}. Input: {input_shape}, Output: {output.shape}, Expected: {expected_output_shape}"
    )
