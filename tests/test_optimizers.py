import numpy as np
import pytest

from mpneuralnetwork.optimizers import SGD, Adam, Optimizer, RMSprop


class MockTrainableLayer:
    """
    A mock layer with trainable parameters that exposes them via a `params` property,
    as expected by the refactored optimizers.
    """

    def __init__(self):
        self.weights = np.ones((10, 5))
        self.biases = np.ones((1, 5))
        self.weights_gradient = np.full_like(self.weights, 0.5)
        self.biases_gradient = np.full_like(self.biases, 0.2)

    @property
    def params(self):
        """Exposes parameters and their corresponding gradients."""
        return {
            "weights": (self.weights, self.weights_gradient),
            "biases": (self.biases, self.biases_gradient),
        }


class MockNonTrainableLayer:
    """A mock layer with no trainable parameters, like an activation function."""

    pass


def test_sgd_with_momentum_updates_parameters():
    """
    Tests that the SGD optimizer (which now includes momentum) correctly updates parameters
    over two steps to verify the momentum calculation.
    """
    # 1. Arrange
    learning_rate = 0.1
    momentum = 0.9
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)

    trainable_layer = MockTrainableLayer()
    layers_list = [trainable_layer]

    # Keep original copies to check updates
    original_weights = np.copy(trainable_layer.weights)
    original_biases = np.copy(trainable_layer.biases)

    # 2. Act (first step)
    optimizer.step(layers_list)

    # 3. Assert (first step)
    # v_1 = momentum * v_0 - lr * grad  (where v_0 is 0)
    # v_1 = -lr * grad
    # W_1 = W_0 + v_1
    velocity_w1 = -learning_rate * trainable_layer.weights_gradient
    expected_weights1 = original_weights + velocity_w1
    assert np.allclose(trainable_layer.weights, expected_weights1), "SGD with Momentum failed on weights (step 1)"

    velocity_b1 = -learning_rate * trainable_layer.biases_gradient
    expected_biases1 = original_biases + velocity_b1
    assert np.allclose(trainable_layer.biases, expected_biases1), "SGD with Momentum failed on biases (step 1)"

    # 4. Act (second step)
    optimizer.step(layers_list)

    # 5. Assert (second step)
    # v_2 = momentum * v_1 - lr * grad
    # W_2 = W_1 + v_2
    velocity_w2 = momentum * velocity_w1 - learning_rate * trainable_layer.weights_gradient
    expected_weights2 = expected_weights1 + velocity_w2
    assert np.allclose(trainable_layer.weights, expected_weights2), "SGD with Momentum failed on weights (step 2)"

    velocity_b2 = momentum * velocity_b1 - learning_rate * trainable_layer.biases_gradient
    expected_biases2 = expected_biases1 + velocity_b2
    assert np.allclose(trainable_layer.biases, expected_biases2), "SGD with Momentum failed on biases (step 2)"


def test_optimizer_handles_non_trainable_layers():
    """
    Tests that optimizers correctly skip layers that do not have
    trainable parameters.
    """
    # Arrange
    optimizer = SGD()  # Any optimizer will do
    layers_list = [MockTrainableLayer(), MockNonTrainableLayer()]

    # Act & Assert: The step should execute without raising an error.
    try:
        optimizer.step(layers_list)
    except Exception as e:
        pytest.fail(f"Optimizer failed on a mixed list of layers with error: {e}")


def test_rmsprop_optimizer_updates_parameters():
    """Tests that the RMSprop optimizer correctly updates parameters."""
    # 1. Arrange
    learning_rate = 0.001
    decay_rate = 0.9
    epsilon = 1e-8
    optimizer = RMSprop(learning_rate=learning_rate, decay_rate=decay_rate, epsilon=epsilon)

    trainable_layer = MockTrainableLayer()
    layers_list = [trainable_layer]

    original_weights = np.copy(trainable_layer.weights)
    grad_w = trainable_layer.weights_gradient

    # 2. Act
    optimizer.step(layers_list)

    # 3. Assert
    # Manually calculate the expected update for weights
    cache_w = (1 - decay_rate) * np.power(grad_w, 2)
    expected_weights = original_weights - learning_rate * grad_w / (np.sqrt(cache_w) + epsilon)
    assert np.allclose(trainable_layer.weights, expected_weights), "RMSprop did not update weights correctly"


def test_adam_optimizer_updates_parameters():
    """Tests that the Adam optimizer correctly updates parameters."""
    # 1. Arrange
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    optimizer = Adam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)

    trainable_layer = MockTrainableLayer()
    layers_list = [trainable_layer]

    original_weights = np.copy(trainable_layer.weights)
    grad_w = trainable_layer.weights_gradient

    # 2. Act
    optimizer.step(layers_list)

    # 3. Assert
    # Manually calculate the expected update for weights for t=1
    t = 1
    m_w = (1 - beta1) * grad_w
    v_w = (1 - beta2) * np.power(grad_w, 2)
    m_hat_w = m_w / (1 - beta1**t)
    v_hat_w = v_w / (1 - beta2**t)
    expected_weights = original_weights - learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
    assert np.allclose(trainable_layer.weights, expected_weights), "Adam did not update weights correctly"


def test_optimizer_base_methods():
    """Test get_config and params for base Optimizer."""
    opt = Optimizer()
    assert opt.get_config() == {"type": "Optimizer"}
    assert opt.params == {}
    opt.step([])  # Should do nothing


def test_optimizer_configs():
    """Test get_config and params for concrete optimizers."""
    # SGD
    sgd = SGD(learning_rate=0.1, momentum=0.5)
    config = sgd.get_config()
    assert config["learning_rate"] == 0.1
    assert config["momentum"] == 0.5
    assert "velocities" in sgd.params

    # RMSprop
    rms = RMSprop(learning_rate=0.02, decay_rate=0.8, epsilon=1e-7)
    config = rms.get_config()
    assert config["learning_rate"] == 0.02
    assert config["decay_rate"] == 0.8
    assert config["epsilon"] == 1e-7
    assert "cache" in rms.params

    # Adam
    adam = Adam(learning_rate=0.03, beta1=0.8, beta2=0.9, epsilon=1e-6)
    config = adam.get_config()
    assert config["learning_rate"] == 0.03
    assert config["beta1"] == 0.8
    assert config["beta2"] == 0.9
    assert config["epsilon"] == 1e-6
    params = adam.params
    assert "t" in params
    assert "momentums" in params
    assert "velocities" in params
