import numpy as np

from mpneuralnetwork.losses import MSE


def test_mse_zero_loss():
    """Tests that MSE loss is zero for identical prediction and target."""
    mse = MSE()
    y = np.array([[1.0, 2.0, 3.0]])
    assert np.isclose(mse.direct(y, y), 0)


def test_mse_value_and_gradient():
    loss_fn = MSE()
    y_pred = np.array([[1.0, 2.0, 3.0]])
    y_true = np.array([[1.5, 2.5, 3.5]])

    expected_loss = np.sum(np.power([-0.5, -0.5, -0.5], 2))
    expected_prime = 2 * np.array([[-0.5, -0.5, -0.5]])

    actual_loss = loss_fn.direct(y_pred, y_true)
    assert np.isclose(actual_loss, expected_loss)

    batch_size = y_pred.shape[0]
    actual_prime = loss_fn.prime(y_pred, y_true)
    assert np.allclose(actual_prime, expected_prime / batch_size, atol=1e-5)
