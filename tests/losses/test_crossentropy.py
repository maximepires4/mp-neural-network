import numpy as np

from mpneuralnetwork.losses import BinaryCrossEntropy, CategoricalCrossEntropy


def test_bce_value_and_gradient():
    loss_fn = BinaryCrossEntropy()
    y_pred = np.array([[-1.0, 1.0, 0.0]])  # Logits
    y_true = np.array([[0.0, 1.0, 1.0]])

    expected_loss = 1.31967055
    # Sigmoid(y_pred) - y_true
    expected_prime = np.array([[(1 / (1 + np.exp(1))) - 0, (1 / (1 + np.exp(-1))) - 1, 0.5 - 1]])

    actual_loss = loss_fn.direct(y_pred, y_true)
    assert np.isclose(actual_loss, expected_loss)

    batch_size = y_pred.shape[0]
    actual_prime = loss_fn.prime(y_pred, y_true)
    assert np.allclose(actual_prime, expected_prime / batch_size, atol=1e-5)


def test_bce_stability_large_logits():
    """
    Tests numerical stability of BCE with Logits.
    Large positive logit -> Sigmoid ~ 1.
    Large negative logit -> Sigmoid ~ 0.
    """
    loss_fn = BinaryCrossEntropy()

    # Case 1: Logit = 100 (Pred ~ 1), True = 1. Loss should be ~0.
    y_pred = np.array([[100.0]])
    y_true = np.array([[1.0]])
    loss = loss_fn.direct(y_pred, y_true)
    assert np.isclose(loss, 0.0)

    # Case 2: Logit = -100 (Pred ~ 0), True = 0. Loss should be ~0.
    y_pred = np.array([[-100.0]])
    y_true = np.array([[0.0]])
    loss = loss_fn.direct(y_pred, y_true)
    assert np.isclose(loss, 0.0)

    # Case 3: Logit = 100 (Pred ~ 1), True = 0. Loss should be huge (linear with x).
    # Loss = max(x, 0) - x*y + log(1 + exp(-abs(x)))
    #      = 100 - 0 + log(1 + 0) = 100.
    y_pred = np.array([[100.0]])
    y_true = np.array([[0.0]])
    loss = loss_fn.direct(y_pred, y_true)
    assert np.isclose(loss, 100.0)


def test_cce_value_and_gradient():
    loss_fn = CategoricalCrossEntropy()
    y_pred = np.array([[0.1, 0.2, 0.7]])  # Logits
    y_true = np.array([[0.0, 0.0, 1.0]])

    expected_loss = 0.76794954
    expected_prime = np.array([[0.25463, 0.28140, -0.53604]])

    actual_loss = loss_fn.direct(y_pred, y_true)
    assert np.isclose(actual_loss, expected_loss)

    batch_size = y_pred.shape[0]
    actual_prime = loss_fn.prime(y_pred, y_true)
    assert np.allclose(actual_prime, expected_prime / batch_size, atol=1e-5)


def test_cce_perfect_prediction():
    cce = CategoricalCrossEntropy()
    logits_perfect = np.array([[-10.0, -10.0, 20.0]])
    y_true_perfect = np.array([[0.0, 0.0, 1.0]])
    assert np.isclose(cce.direct(logits_perfect, y_true_perfect), 0)
