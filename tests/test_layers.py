import numpy as np
import pytest
from mpneuralnetwork.layers import Dense
from mpneuralnetwork.losses import MSE


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
    layer = Dense(n_inputs, n_outputs)
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
    layer = Dense(n_inputs, n_outputs)
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