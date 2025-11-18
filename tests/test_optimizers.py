import numpy as np
import pytest
from mpneuralnetwork.optimizers import SGD


class MockTrainableLayer:
    """A mock layer with trainable parameters (weights and biases) to test the optimizer."""
    def __init__(self, weights_shape=(10, 5), biases_shape=(1, 5)):
        self.weights = np.ones(weights_shape)
        self.biases = np.ones(biases_shape)
        self.weights_gradient = np.full(weights_shape, 0.5)
        self.biases_gradient = np.full(biases_shape, 0.2)
        # The optimizer should ignore parameters that are None
        self.kernels = None
        self.kernels_gradient = None


class MockNonTrainableLayer:
    """A mock layer with no trainable parameters, like an activation function."""
    pass


def test_sgd_optimizer_updates_parameters():
    """
    Tests that the SGD optimizer correctly updates layer parameters according to its rule.
    """
    # 1. Arrange
    learning_rate = 0.1
    optimizer = SGD(learning_rate=learning_rate)
    
    trainable_layer = MockTrainableLayer()
    layers_list = [trainable_layer]

    # Store original parameters to verify the update
    original_weights = np.copy(trainable_layer.weights)
    original_biases = np.copy(trainable_layer.biases)

    # 2. Act
    optimizer.step(layers_list)

    # 3. Assert
    # Check if weights were updated correctly: new = old - lr * grad
    expected_weights = original_weights - learning_rate * trainable_layer.weights_gradient
    assert np.allclose(trainable_layer.weights, expected_weights), "SGD did not update weights correctly"
    
    # Check if biases were updated correctly
    expected_biases = original_biases - learning_rate * trainable_layer.biases_gradient
    assert np.allclose(trainable_layer.biases, expected_biases), "SGD did not update biases correctly"


def test_sgd_optimizer_handles_non_trainable_layers():
    """
    Tests that the SGD optimizer runs without error on layers that do not have
    trainable parameters (e.g., activation layers).
    """
    # 1. Arrange
    optimizer = SGD(learning_rate=0.1)
    
    # Create a list of layers including trainable and non-trainable ones
    layers_list = [MockTrainableLayer(), MockNonTrainableLayer()]

    # 2. Act & Assert
    # The step should execute without raising an AttributeError or any other error
    try:
        optimizer.step(layers_list)
    except Exception as e:
        pytest.fail(f"SGD optimizer failed on a list of mixed layer types with error: {e}")