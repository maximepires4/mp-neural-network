import numpy as np
import pytest
from mpneuralnetwork.optimizers import SGD, SGDMomentum, RMSprop, Adam


class MockTrainableLayer:
    """A mock layer with trainable parameters to test the optimizer."""
    def __init__(self, weights_shape=(10, 5), biases_shape=(1, 5), kernels_shape=(3, 3, 1, 4)):
        self.weights = np.ones(weights_shape)
        self.biases = np.ones(biases_shape)
        self.kernels = np.ones(kernels_shape)
        self.weights_gradient = np.full(weights_shape, 0.5)
        self.biases_gradient = np.full(biases_shape, 0.2)
        self.kernels_gradient = np.full(kernels_shape, 0.3)


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
    original_kernels = np.copy(trainable_layer.kernels)

    # 2. Act
    optimizer.step(layers_list)

    # 3. Assert
    # Check if weights were updated correctly: new = old - lr * grad
    expected_weights = original_weights - learning_rate * trainable_layer.weights_gradient
    assert np.allclose(trainable_layer.weights, expected_weights), "SGD did not update weights correctly"
    
    # Check if biases were updated correctly
    expected_biases = original_biases - learning_rate * trainable_layer.biases_gradient
    assert np.allclose(trainable_layer.biases, expected_biases), "SGD did not update biases correctly"

    # Check if kernels were updated correctly
    expected_kernels = original_kernels - learning_rate * trainable_layer.kernels_gradient
    assert np.allclose(trainable_layer.kernels, expected_kernels), "SGD did not update kernels correctly"


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


def test_sgdmomentum_optimizer_updates_parameters():
    """
    Tests that the SGDMomentum optimizer correctly updates layer parameters.
    """
    # Arrange
    learning_rate = 0.1
    momentum = 0.9
    optimizer = SGDMomentum(learning_rate=learning_rate, momentum=momentum)
    
    trainable_layer = MockTrainableLayer()
    layers_list = [trainable_layer]

    original_weights = np.copy(trainable_layer.weights)
    original_biases = np.copy(trainable_layer.biases)
    original_kernels = np.copy(trainable_layer.kernels)

    # Act (first step)
    optimizer.step(layers_list)

    # Assert (first step)
    # Weights
    velocity_w1 = -learning_rate * trainable_layer.weights_gradient
    expected_weights1 = original_weights + velocity_w1
    assert np.allclose(trainable_layer.weights, expected_weights1), "SGDMomentum did not update weights correctly on first step"
    # Biases
    velocity_b1 = -learning_rate * trainable_layer.biases_gradient
    expected_biases1 = original_biases + velocity_b1
    assert np.allclose(trainable_layer.biases, expected_biases1), "SGDMomentum did not update biases correctly on first step"
    # Kernels
    velocity_k1 = -learning_rate * trainable_layer.kernels_gradient
    expected_kernels1 = original_kernels + velocity_k1
    assert np.allclose(trainable_layer.kernels, expected_kernels1), "SGDMomentum did not update kernels correctly on first step"

    # Act (second step to check momentum)
    optimizer.step(layers_list)

    # Assert (second step)
    # Weights
    velocity_w2 = momentum * velocity_w1 - learning_rate * trainable_layer.weights_gradient
    expected_weights2 = expected_weights1 + velocity_w2
    assert np.allclose(trainable_layer.weights, expected_weights2), "SGDMomentum did not update weights correctly on second step"
    # Biases
    velocity_b2 = momentum * velocity_b1 - learning_rate * trainable_layer.biases_gradient
    expected_biases2 = expected_biases1 + velocity_b2
    assert np.allclose(trainable_layer.biases, expected_biases2), "SGDMomentum did not update biases correctly on second step"
    # Kernels
    velocity_k2 = momentum * velocity_k1 - learning_rate * trainable_layer.kernels_gradient
    expected_kernels2 = expected_kernels1 + velocity_k2
    assert np.allclose(trainable_layer.kernels, expected_kernels2), "SGDMomentum did not update kernels correctly on second step"


def test_rmsprop_optimizer_updates_parameters():
    """
    Tests that the RMSprop optimizer correctly updates layer parameters.
    """
    # Arrange
    learning_rate = 0.001
    decay_rate = 0.9
    epsilon = 1e-8
    optimizer = RMSprop(learning_rate=learning_rate, decay_rate=decay_rate, epsilon=epsilon)
    
    trainable_layer = MockTrainableLayer()
    layers_list = [trainable_layer]

    original_weights = np.copy(trainable_layer.weights)
    original_biases = np.copy(trainable_layer.biases)
    original_kernels = np.copy(trainable_layer.kernels)

    # Act
    optimizer.step(layers_list)

    # Assert
    # Weights
    cache_w = (1 - decay_rate) * np.power(trainable_layer.weights_gradient, 2)
    expected_weights = original_weights - learning_rate * trainable_layer.weights_gradient / (np.sqrt(cache_w) + epsilon)
    assert np.allclose(trainable_layer.weights, expected_weights), "RMSprop did not update weights correctly"
    
    # Biases
    cache_b = (1 - decay_rate) * np.power(trainable_layer.biases_gradient, 2)
    expected_biases = original_biases - learning_rate * trainable_layer.biases_gradient / (np.sqrt(cache_b) + epsilon)
    assert np.allclose(trainable_layer.biases, expected_biases), "RMSprop did not update biases correctly"

    # Kernels
    cache_k = (1 - decay_rate) * np.power(trainable_layer.kernels_gradient, 2)
    expected_kernels = original_kernels - learning_rate * trainable_layer.kernels_gradient / (np.sqrt(cache_k) + epsilon)
    assert np.allclose(trainable_layer.kernels, expected_kernels), "RMSprop did not update kernels correctly"


def test_adam_optimizer_updates_parameters():
    """
    Tests that the Adam optimizer correctly updates layer parameters.
    """
    # Arrange
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    optimizer = Adam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    
    trainable_layer = MockTrainableLayer()
    layers_list = [trainable_layer]

    original_weights = np.copy(trainable_layer.weights)
    original_biases = np.copy(trainable_layer.biases)
    original_kernels = np.copy(trainable_layer.kernels)

    # Act
    optimizer.step(layers_list)

    # Assert
    t = 1
    # Weights
    grad_w = trainable_layer.weights_gradient
    m_w = (1 - beta1) * grad_w
    v_w = (1 - beta2) * np.power(grad_w, 2)
    m_hat_w = m_w / (1 - beta1**t)
    v_hat_w = v_w / (1 - beta2**t)
    expected_weights = original_weights - learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
    assert np.allclose(trainable_layer.weights, expected_weights), "Adam did not update weights correctly"
    
    # Biases
    grad_b = trainable_layer.biases_gradient
    m_b = (1 - beta1) * grad_b
    v_b = (1 - beta2) * np.power(grad_b, 2)
    m_hat_b = m_b / (1 - beta1**t)
    v_hat_b = v_b / (1 - beta2**t)
    expected_biases = original_biases - learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
    assert np.allclose(trainable_layer.biases, expected_biases), "Adam did not update biases correctly"

    # Kernels
    grad_k = trainable_layer.kernels_gradient
    m_k = (1 - beta1) * grad_k
    v_k = (1 - beta2) * np.power(grad_k, 2)
    m_hat_k = m_k / (1 - beta1**t)
    v_hat_k = v_k / (1 - beta2**t)
    expected_kernels = original_kernels - learning_rate * m_hat_k / (np.sqrt(v_hat_k) + epsilon)
    assert np.allclose(trainable_layer.kernels, expected_kernels), "Adam did not update kernels correctly"
