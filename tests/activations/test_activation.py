import numpy as np
import pytest

from mpneuralnetwork.activations import PReLU, ReLU, Sigmoid, Softmax, Swish, Tanh
from mpneuralnetwork.losses import MSE
from tests.utils import check_gradient


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
    np.random.seed(69)
    batch_size, n_inputs = 4, 5

    layer = activation_class(**activation_args)
    loss_fn = MSE()

    X = np.random.randn(batch_size, n_inputs)
    Y = np.random.randn(batch_size, n_inputs)

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
    layer = activation_class(**activation_args)
    input_shape = (64, 10)
    input_data = np.random.randn(*input_shape)
    output = layer.forward(input_data)
    assert output.shape == input_shape
