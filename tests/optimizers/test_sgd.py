import numpy as np

from mpneuralnetwork.optimizers import SGD
from tests.optimizers.test_optimizers import MockTrainableLayer


def test_sgd_with_momentum_updates_parameters(mock_trainable_layer: MockTrainableLayer):
    learning_rate = 0.1
    momentum = 0.9
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    layers_list = [mock_trainable_layer]

    original_weights = np.copy(mock_trainable_layer.weights)
    original_biases = np.copy(mock_trainable_layer.biases)

    # Step 1
    optimizer.step(layers_list)

    velocity_w1 = -learning_rate * mock_trainable_layer.weights_gradient
    expected_weights1 = original_weights + velocity_w1
    assert np.allclose(mock_trainable_layer.weights, expected_weights1)

    velocity_b1 = -learning_rate * mock_trainable_layer.biases_gradient
    expected_biases1 = original_biases + velocity_b1
    assert np.allclose(mock_trainable_layer.biases, expected_biases1)

    # Step 2
    optimizer.step(layers_list)

    velocity_w2 = momentum * velocity_w1 - learning_rate * mock_trainable_layer.weights_gradient
    expected_weights2 = expected_weights1 + velocity_w2
    assert np.allclose(mock_trainable_layer.weights, expected_weights2)
