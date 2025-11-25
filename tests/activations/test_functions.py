import numpy as np
import pytest

from mpneuralnetwork.activations import PReLU, ReLU, Sigmoid, Softmax, Swish, Tanh


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
    activation = Sigmoid()
    assert np.allclose(activation.forward(input_val), expected_forward)
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
    activation = PReLU(alpha=0.01)
    assert np.allclose(activation.forward(input_val), expected_forward)
    activation.forward(input_val)
    assert np.allclose(activation.backward(1), expected_backward)


@pytest.mark.parametrize(
    "input_val, expected_forward, expected_backward",
    [
        (0, 0, 0.5),
        (1, 0.73105858, 0.9276712),
    ],
)
def test_swish(input_val, expected_forward, expected_backward):
    activation = Swish()
    assert np.allclose(activation.forward(input_val), expected_forward)
    activation.forward(input_val)
    assert np.allclose(activation.backward(1), expected_backward)


def test_prelu_config():
    prelu = PReLU(alpha=0.25)
    config = prelu.get_config()
    assert config["alpha"] == 0.25
    assert config["type"] == "PReLU"


def test_softmax_params():
    softmax = Softmax()
    assert softmax.params == {}
