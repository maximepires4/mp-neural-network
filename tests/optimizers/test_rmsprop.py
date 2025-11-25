import numpy as np

from mpneuralnetwork.optimizers import RMSprop
from tests.optimizers.test_optimizers import MockTrainableLayer


def test_rmsprop_optimizer_updates_parameters(mock_trainable_layer: MockTrainableLayer):
    learning_rate = 0.001
    decay_rate = 0.9
    epsilon = 1e-8
    optimizer = RMSprop(learning_rate=learning_rate, decay_rate=decay_rate, epsilon=epsilon)
    layers_list = [mock_trainable_layer]

    original_weights = np.copy(mock_trainable_layer.weights)
    grad_w = mock_trainable_layer.weights_gradient

    optimizer.step(layers_list)

    cache_w = (1 - decay_rate) * np.power(grad_w, 2)
    expected_weights = original_weights - learning_rate * grad_w / (np.sqrt(cache_w) + epsilon)
    assert np.allclose(mock_trainable_layer.weights, expected_weights)
