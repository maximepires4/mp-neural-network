import numpy as np
import pytest

from mpneuralnetwork.activations import PReLU, ReLU, Sigmoid, Softmax, Swish, Tanh
from mpneuralnetwork.losses import MSE
from tests.utils import check_gradient


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
        (1, 0.73105858, 0.9276712),  # f'(1) = f(1) + sigmoid(1)*(1-f(1))
    ],
)
def test_swish(input_val, expected_forward, expected_backward):
    """Tests the Swish activation for various inputs."""
    activation = Swish()
    assert np.allclose(activation.forward(input_val), expected_forward)
    activation.forward(input_val)
    assert np.allclose(activation.backward(1), expected_backward)


@pytest.mark.parametrize(
    "activation_class, activation_args",
    [
        (Sigmoid, {}),
        (Tanh, {}),
        (ReLU, {}),
        (PReLU, {"alpha": 0.01}),
        (Swish, {}),
        (Softmax, {}),
    ],
)
def test_activation_gradients(activation_class, activation_args):
    """
    Performs numerical gradient checking for all activation layers.
    """
    np.random.seed(69)
    batch_size, n_inputs = 4, 5

    layer = activation_class(**activation_args)
    loss_fn = MSE()

    X = np.random.randn(batch_size, n_inputs)
    Y = np.random.randn(batch_size, n_inputs)

    # PReLU can be unstable with large inputs, so we scale them down for this test
    if isinstance(layer, PReLU):
        X /= 10

    check_gradient(layer, X, Y, loss_fn)


@pytest.mark.parametrize(
    "activation_class, activation_args",
    [
        (Sigmoid, {}),
        (Tanh, {}),
        (ReLU, {}),
        (PReLU, {"alpha": 0.01}),
        (Swish, {}),
        (Softmax, {}),
    ],
)
def test_activation_output_shapes(activation_class, activation_args):
    """
    Tests that the output shape of an activation function's forward pass is the same as the input shape.
    """
    # 1. Arrange
    layer = activation_class(**activation_args)
    input_shape = (64, 10)  # Example batch of 64 samples, 10 features
    input_data = np.random.randn(*input_shape)

    # 2. Act
    output = layer.forward(input_data)

    # 3. Assert
    assert output.shape == input_shape, f"Shape mismatch for activation {activation_class.__name__}. Input: {input_shape}, Output: {output.shape}"


def test_prelu_config():
    """Test get_config for PReLU."""
    prelu = PReLU(alpha=0.25)
    config = prelu.get_config()
    assert config["alpha"] == 0.25
    assert config["type"] == "PReLU"


def test_softmax_params():
    """Test params property for Softmax."""
    softmax = Softmax()
    assert softmax.params == {}
