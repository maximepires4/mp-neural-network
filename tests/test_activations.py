import numpy as np
import pytest
from mpneuralnetwork.activations import Sigmoid, Tanh, ReLU, PReLU, Swish, Softmax
from mpneuralnetwork.losses import MSE


@pytest.mark.parametrize(
    "input_val, expected_forward, expected_backward",
    [
        (0, 0.5, 0.25),
        (1, 0.73105858, 0.19661193),
        (-1, 0.26894142, 0.19661193),
        (np.array([-2, 0, 2]), np.array([0.11920292, 0.5, 0.88079708]), np.array([0.10499359, 0.25, 0.10499359])),
    ],
)
def test_sigmoid(input_val, expected_forward, expected_backward):
    """Tests the Sigmoid activation for various inputs."""
    activation = Sigmoid()
    # Test forward pass
    assert np.allclose(activation.forward(input_val), expected_forward)
    # Test backward pass (must be called after forward)
    assert np.allclose(activation.backward(1), expected_backward)


@pytest.mark.parametrize(
    "input_val, expected_forward, expected_backward",
    [
        (0, 0, 1),
        (1, 0.76159416, 0.41997434),
        (-1, -0.76159416, 0.41997434),
        (np.array([-2, 0, 2]), np.array([-0.96402758, 0, 0.96402758]), np.array([0.07065082, 1, 0.07065082])),
    ],
)
def test_tanh(input_val, expected_forward, expected_backward):
    """Tests the Tanh activation for various inputs."""
    activation = Tanh()
    assert np.allclose(activation.forward(input_val), expected_forward)
    activation.forward(input_val)
    assert np.allclose(activation.backward(1), expected_backward)


@pytest.mark.parametrize(
    "input_val, expected_forward, expected_backward",
    [
        (10, 10, 1),
        (-10, 0, 0),
        (0, 0, 0),
        (np.array([-5, 0, 5]), np.array([0, 0, 5]), np.array([0, 0, 1])),
    ],
)
def test_relu(input_val, expected_forward, expected_backward):
    """Tests the ReLU activation for various inputs."""
    activation = ReLU()
    assert np.allclose(activation.forward(input_val), expected_forward)
    activation.forward(input_val)
    assert np.allclose(activation.backward(1), expected_backward)


@pytest.mark.parametrize(
    "input_val, expected_forward, expected_backward",
    [
        (10, 10, 1),
        (-10, -0.1, 0.01),
        (np.array([-5, 0, 5]), np.array([-0.05, 0, 5]), np.array([0.01, 1, 1])),
    ],
)
def test_prelu(input_val, expected_forward, expected_backward):
    """Tests the PReLU activation with default alpha=0.01."""
    activation = PReLU(alpha=0.01)
    assert np.allclose(activation.forward(input_val), expected_forward)
    activation.forward(input_val)
    assert np.allclose(activation.backward(1), expected_backward)


@pytest.mark.parametrize(
    "input_val, expected_forward, expected_backward",
    [
        (0, 0, 0.5),
        (1, 0.73105858, 0.9276712), # f'(1) = f(1) + sigmoid(1)*(1-f(1))
    ],
)
def test_swish(input_val, expected_forward, expected_backward):
    """Tests the Swish activation for various inputs."""
    activation = Swish()
    assert np.allclose(activation.forward(input_val), expected_forward)
    activation.forward(input_val)
    assert np.allclose(activation.backward(1), expected_backward)


def test_softmax_gradient_checking():
    """
    Performs numerical gradient checking for the Softmax layer's backward pass.
    This is a more complex test to ensure mathematical correctness of the gradient.
    """
    # 1. Setup
    batch_size, n_inputs = 4, 5
    layer = Softmax()
    loss_fn = MSE()
    epsilon = 1e-5

    # 2. Random data
    X = np.random.randn(batch_size, n_inputs)
    Y = np.random.randn(batch_size, n_inputs)

    # 3. Calculate analytical gradient (the one computed by the function)
    preds = layer.forward(X)
    output_gradient = loss_fn.prime(preds, Y)
    analytical_grads_x = layer.backward(output_gradient)

    # 4. Calculate numerical gradient (the "true" gradient)
    numerical_grads_x = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Calculate loss for x + epsilon
            X[i, j] += epsilon
            preds_plus = layer.forward(X)
            loss_plus = loss_fn.direct(preds_plus, Y)
            
            # Calculate loss for x - epsilon
            X[i, j] -= 2 * epsilon
            preds_minus = layer.forward(X)
            loss_minus = loss_fn.direct(preds_minus, Y)
            
            # Restore original value
            X[i, j] += epsilon
            
            # Compute the slope
            numerical_grads_x[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
    # 5. Assert that the two gradients are close
    assert np.allclose(analytical_grads_x, numerical_grads_x, atol=1e-5), "Softmax input gradients do not match"
