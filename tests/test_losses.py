import numpy as np
import pytest
from mpneuralnetwork.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy


# --- Test Cases ---
# Each tuple contains: (loss_class, y_pred, y_true, expected_loss, expected_prime)

mse_cases = [
    (
        MSE,
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[1.5, 2.5, 3.5]]),
        np.sum(np.power([-0.5, -0.5, -0.5], 2)),  # sum of squares for one sample
        2 * np.array([[-0.5, -0.5, -0.5]]), # prime is not averaged here
    )
]

bce_cases = [
    (
        BinaryCrossEntropy,
        np.array([[-1.0, 1.0, 0.0]]), # logits
        np.array([[0.0, 1.0, 1.0]]),
        1.31967055, # Sum of losses for the sample
        np.array([[(1/(1+np.exp(1))) - 0, (1/(1+np.exp(-1))) - 1, 0.5 - 1]]),
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

@pytest.mark.parametrize(
    "loss_class, y_pred, y_true, expected_loss, expected_prime", all_cases
)
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