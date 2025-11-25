from collections.abc import Iterator

import numpy as np
import pytest

from mpneuralnetwork.optimizers import SGD, Adam, Optimizer, RMSprop


class MockTrainableLayer:
    def __init__(self) -> None:
        self.weights = np.ones((10, 5))
        self.biases = np.ones((1, 5))
        self.weights_gradient = np.full_like(self.weights, 0.5)
        self.biases_gradient = np.full_like(self.biases, 0.2)

    @property
    def params(self) -> dict:
        return {
            "weights": (self.weights, self.weights_gradient),
            "biases": (self.biases, self.biases_gradient),
        }


class MockNonTrainableLayer:
    pass


@pytest.fixture
def mock_trainable_layer() -> Iterator[MockTrainableLayer]:
    yield MockTrainableLayer()


def test_optimizer_base_methods():
    opt = Optimizer(learning_rate=0.1, regularization="L2", weight_decay=0.01)
    expected_config = {
        "type": "Optimizer",
        "learning_rate": 0.1,
        "regularization": "L2",
        "weight_decay": 0.01,
    }
    assert opt.get_config() == expected_config
    assert opt.params == {}
    opt.step([])


def test_optimizer_handles_non_trainable_layers(mock_trainable_layer):
    optimizer = SGD()
    layers_list = [mock_trainable_layer, MockNonTrainableLayer()]
    try:
        optimizer.step(layers_list)
    except Exception as e:
        pytest.fail(f"Optimizer failed on a mixed list of layers with error: {e}")


def test_optimizer_configs():
    sgd = SGD(learning_rate=0.1, momentum=0.5)
    config = sgd.get_config()
    assert config["learning_rate"] == 0.1
    assert config["momentum"] == 0.5
    assert "velocities" in sgd.params

    rms = RMSprop(learning_rate=0.02, decay_rate=0.8, epsilon=1e-7)
    config = rms.get_config()
    assert config["learning_rate"] == 0.02
    assert config["decay_rate"] == 0.8
    assert config["epsilon"] == 1e-7
    assert "cache" in rms.params

    adam = Adam(learning_rate=0.03, beta1=0.8, beta2=0.9, epsilon=1e-6)
    config = adam.get_config()
    assert config["learning_rate"] == 0.03
    assert config["beta1"] == 0.8
    assert config["beta2"] == 0.9
    assert config["epsilon"] == 1e-6
    assert "t" in adam.params
