import numpy as np
import pytest
from mpneuralnetwork.layers import Dense, Dropout
from mpneuralnetwork.losses import MSE


np.random.seed(69)  # For reproducible test data in parametrization


@pytest.mark.parametrize(
    "input_data, weights, biases, expected_shape, expected_output",
    [
        # Case 1: Shape check with a batch
        (np.random.randn(64, 10), np.random.randn(10, 5), np.random.randn(1, 5), (64, 5), None),
        
        # Case 2: Forward pass value check
        (
            np.array([[0.2, 0.8]]),          # input
            np.array([[0.5], [0.5]]),        # weights
            np.array([[0.1]]),              # biases
            (1, 1),                         # expected_shape
            np.array([[0.6]])               # expected_output: (0.2*0.5 + 0.8*0.5) + 0.1 = 0.6
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
    layer = Dense(n_inputs, n_outputs, initialization='xavier')
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
    layer = Dense(n_inputs, n_outputs, initialization='xavier')
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
            
            layer.weights[i, j] += epsilon # Restore
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
        
        layer.biases[0, i] += epsilon # Restore
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
            
            X[i, j] += epsilon # Restore
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
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
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
    assert np.allclose(analytical_grads, numerical_grads, atol=atol), (
        f"Gradient mismatch for layer {layer.__class__.__name__}"
    )




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
    
    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
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


@pytest.mark.parametrize(
    "layer, input_shape, expected_output_shape",
    [
        # Dense Layer
        (Dense(10, 5, initialization='xavier'), (64, 10), (64, 5)),
        (Dense(3, 1, initialization='xavier'), (1, 3), (1, 1)),
        
        # Dropout Layer (should not change shape)
        (Dropout(0.5), (128, 20), (128, 20)),
    ]
)
def test_layer_output_shapes(layer, input_shape, expected_output_shape):
    """
    Tests that the output shape of a layer's forward pass is correct.
    """
    # 1. Arrange: Create random input data with the specified shape
    input_data = np.random.randn(*input_shape)

    # 2. Act: Perform a forward pass
    output = layer.forward(input_data)

    # 3. Assert: Check if the output shape matches the expected shape
    assert output.shape == expected_output_shape, (
        f"Shape mismatch for layer {layer.__class__.__name__}. "
        f"Input: {input_shape}, Output: {output.shape}, Expected: {expected_output_shape}"
    )
    
    
