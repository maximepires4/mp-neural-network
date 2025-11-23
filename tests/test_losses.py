import numpy as np
import pytest

from mpneuralnetwork.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy, Loss


def test_loss_configs():
    """Test get_config for losses."""

    # Base Loss (using a concrete mock implementation)
    class MockLoss(Loss):
        def direct(self, o, e):
            pass

        def prime(self, o, e):
            pass

    loss = MockLoss()
    assert loss.get_config() == {"type": "MockLoss"}

    # MSE
    mse = MSE()
    assert mse.get_config() == {"type": "MSE"}


# --- Test Cases ---
# Each tuple contains: (loss_class, y_pred, y_true, expected_loss, expected_prime)

mse_cases = [
    (
        MSE,
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[1.5, 2.5, 3.5]]),
        np.sum(np.power([-0.5, -0.5, -0.5], 2)),  # sum of squares for one sample
        2 * np.array([[-0.5, -0.5, -0.5]]),  # prime is not averaged here
    )
]

bce_cases = [
    (
        BinaryCrossEntropy,
        np.array([[-1.0, 1.0, 0.0]]),  # logits
        np.array([[0.0, 1.0, 1.0]]),
        1.31967055,  # Sum of losses for the sample
        np.array([[(1 / (1 + np.exp(1))) - 0, (1 / (1 + np.exp(-1))) - 1, 0.5 - 1]]),
    )
]

cce_cases = [
    (
        CategoricalCrossEntropy,
        np.array([[0.1, 0.2, 0.7]]),  # logits
        np.array([[0.0, 0.0, 1.0]]),
        0.76794954,  # Corrected loss for the sample
        np.array([[0.25463, 0.28140, -0.53604]]),  # More precise softmax(logits) - y_true
    )
]

all_cases = mse_cases + bce_cases + cce_cases


@pytest.mark.parametrize("loss_class, y_pred, y_true, expected_loss, expected_prime", all_cases)
def test_loss_functions(loss_class, y_pred, y_true, expected_loss, expected_prime):
    """
    Tests the direct (loss) and prime (gradient) methods for all loss functions.
    Note: The prime method returns a gradient that is averaged over the batch size in the implementation.
    The test cases here use a batch size of 1, so the expected_prime is not divided.
    """
    loss_fn = loss_class()
    batch_size = y_pred.shape[0]

    # 1. Test the direct loss calculation
    # The implementation averages the sum of losses per sample over the batch.
    # Since our batch_size is 1, the result is just the sum.
    actual_loss = loss_fn.direct(y_pred, y_true)
    assert np.isclose(actual_loss, expected_loss), f"{loss_class.__name__} direct loss failed"

    # 2. Test the prime (gradient) calculation
    # The implementation returns the gradient averaged by batch_size.
    actual_prime = loss_fn.prime(y_pred, y_true)
    expected_prime_averaged = expected_prime / batch_size
    assert np.allclose(actual_prime, expected_prime_averaged, atol=1e-5), f"{loss_class.__name__} prime gradient failed"


# Specific test for zero loss
def test_mse_zero_loss():
    """Tests that MSE loss is zero for identical prediction and target."""
    mse = MSE()
    y = np.array([[1.0, 2.0, 3.0]])
    assert np.isclose(mse.direct(y, y), 0)


def test_cce_perfect_prediction():
    """Tests that CCE loss is near zero for a perfect prediction."""

    cce = CategoricalCrossEntropy()

    logits_perfect = np.array([[-10.0, -10.0, 20.0]])  # Almost certain prediction

    y_true_perfect = np.array([[0.0, 0.0, 1.0]])

    assert np.isclose(cce.direct(logits_perfect, y_true_perfect), 0)


@pytest.mark.parametrize(
    "loss_class, y_pred_shape, y_true_shape",
    [
        (MSE, (64, 10), (64, 10)),
        (BinaryCrossEntropy, (32, 1), (32, 1)),
        (CategoricalCrossEntropy, (16, 5), (16, 5)),
    ],
)
def test_loss_gradient_shape(loss_class, y_pred_shape, y_true_shape):
    """

    Tests that the shape of the gradient from the loss's prime method

    matches the shape of the input predictions.

    """

    # 1. Arrange

    loss_fn = loss_class()

    y_pred = np.random.randn(*y_pred_shape)

    y_true = np.random.randn(*y_true_shape)

    # 2. Act

    gradient = loss_fn.prime(y_pred, y_true)

    # 3. Assert

    assert gradient.shape == y_pred.shape, f"Shape mismatch for {loss_class.__name__} gradient. Expected {y_pred.shape}, but got {gradient.shape}."


def _check_loss_gradient(loss_fn, y_pred, y_true, epsilon=1e-5, atol=1e-5):
    """

    Helper function to perform numerical gradient checking for a loss function's prime method.

    """

    # 1. Analytical gradient

    analytical_grad = loss_fn.prime(y_pred.copy(), y_true)

    # 2. Numerical gradient

    numerical_grad = np.zeros_like(y_pred)

    it = np.nditer(y_pred, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        ix = it.multi_index

        original_value = y_pred[ix]

        # Loss for y_pred + epsilon

        y_pred[ix] = original_value + epsilon

        loss_plus = loss_fn.direct(y_pred.copy(), y_true)

        # Loss for y_pred - epsilon

        y_pred[ix] = original_value - epsilon

        loss_minus = loss_fn.direct(y_pred.copy(), y_true)

        # Restore original value

        y_pred[ix] = original_value

        # Compute numerical gradient

        numerical_grad[ix] = (loss_plus - loss_minus) / (2 * epsilon)

        it.iternext()

    # 3. Assert

    assert np.allclose(analytical_grad, numerical_grad, atol=atol), f"Gradient mismatch for loss {loss_fn.__class__.__name__}"


@pytest.mark.parametrize(
    "loss_class, y_pred_shape, y_true_shape",
    [
        (MSE, (4, 5), (4, 5)),
        (BinaryCrossEntropy, (8, 1), (8, 1)),
        (CategoricalCrossEntropy, (16, 10), (16, 10)),
    ],
)
def test_loss_numerical_gradients(loss_class, y_pred_shape, y_true_shape):
    """

    Performs numerical gradient checking for all loss functions.

    """

    np.random.seed(69)

    loss_fn = loss_class()

    y_pred = np.random.randn(*y_pred_shape)

    y_true = np.random.randn(*y_true_shape)

    # For CCE, y_true should be one-hot encoded

    if isinstance(loss_fn, CategoricalCrossEntropy):
        y_true = np.eye(y_true_shape[1])[np.random.choice(y_true_shape[1], y_true_shape[0])]

    # For BCE, y_true should be 0s and 1s

    if isinstance(loss_fn, BinaryCrossEntropy):
        y_true = np.random.randint(0, 2, size=y_true_shape)

    _check_loss_gradient(loss_fn, y_pred, y_true)
