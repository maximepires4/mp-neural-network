import numpy as np

from mpneuralnetwork.optimizers import Adam
from tests.optimizers.test_optimizers import MockTrainableLayer


def test_adam_optimizer_updates_parameters(mock_trainable_layer: MockTrainableLayer):
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    optimizer = Adam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    layers_list = [mock_trainable_layer]

    original_weights = np.copy(mock_trainable_layer.weights)
    grad_w = mock_trainable_layer.weights_gradient

    optimizer.step(layers_list)

    t = 1
    m_w = (1 - beta1) * grad_w
    v_w = (1 - beta2) * np.power(grad_w, 2)
    m_hat_w = m_w / (1 - beta1**t)
    v_hat_w = v_w / (1 - beta2**t)
    expected_weights = original_weights - learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
    assert np.allclose(mock_trainable_layer.weights, expected_weights)


def test_adam_l2_regularization_behaves_like_adamw(mock_trainable_layer: MockTrainableLayer):
    learning_rate = 0.1
    weight_decay = 0.1
    optimizer = Adam(learning_rate=learning_rate, regularization="L2", weight_decay=weight_decay)

    original_weights = np.copy(mock_trainable_layer.weights)
    mock_trainable_layer.weights_gradient.fill(0.0)
    mock_trainable_layer.biases_gradient.fill(0.0)

    optimizer.step([mock_trainable_layer])

    expected_factor = 1 - learning_rate * weight_decay
    expected_weights = original_weights * expected_factor
    assert np.allclose(mock_trainable_layer.weights, expected_weights)


def test_adam_l1_regularization_affects_gradients(mock_trainable_layer: MockTrainableLayer):
    learning_rate = 0.1
    weight_decay = 0.5
    optimizer = Adam(learning_rate=learning_rate, regularization="L1", weight_decay=weight_decay)

    mock_trainable_layer.weights_gradient.fill(0.0)
    optimizer.step([mock_trainable_layer])

    assert not np.allclose(mock_trainable_layer.weights, 1.0)
    assert np.all(mock_trainable_layer.weights < 1.0)
