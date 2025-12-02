import numpy as np

from mpneuralnetwork.activations import Softmax
from mpneuralnetwork.losses import CategoricalCrossEntropy


def test_softmax_temperature_smoothing():
    """Test that higher temperature smoothes the output distribution."""
    logits = np.array([[10.0, 5.0, 2.0]])

    # T=1.0 (Standard)
    s1 = Softmax(temperature=1.0)
    out1 = s1.forward(logits)

    # T=5.0 (High Temp -> Smoother)
    s2 = Softmax(temperature=5.0)
    out2 = s2.forward(logits)

    # The highest probability should decrease with higher temperature
    assert np.max(out2) < np.max(out1)

    # The lowest probability should increase (distribution gets closer to uniform)
    assert np.min(out2) > np.min(out1)

    # Sum must still be 1
    assert np.isclose(np.sum(out2), 1.0)


def test_softmax_temperature_sharpening():
    """Test that lower temperature sharpens the output distribution."""
    logits = np.array([[0.6, 0.4, 0.1]])

    # T=1.0
    s1 = Softmax(temperature=1.0)
    out1 = s1.forward(logits)

    # T=0.1 (Low Temp -> Sharper)
    s2 = Softmax(temperature=0.1)
    out2 = s2.forward(logits)

    # The highest probability should increase (closer to 1)
    assert np.max(out2) > np.max(out1)

    # Ideally closer to argmax
    assert np.argmax(out2) == np.argmax(logits)


def test_cce_gradient_scaling_with_temperature():
    """Test that the gradient of CCE is correctly scaled by 1/T."""
    # Setup
    logits = np.array([[2.0, 1.0, 0.1]])
    target = np.array([[1.0, 0.0, 0.0]])  # True class is index 0

    # Case 2: T=2.0
    loss2 = CategoricalCrossEntropy(temperature=2.0)
    grad2 = loss2.prime(logits, target)

    # Manual check:
    # Softmax(logits/1) = [0.705, 0.259, 0.035]
    # Softmax(logits/2) = [0.532, 0.323, 0.145]

    # Gradient formula: (Softmax(z/T) - y) / T

    # We can't just compare grad1 / 2 because the Softmax values themselves change with T.
    # Instead, let's verify the mathematical consistency for T=2

    # Calculate expected gradient manually
    # 1. Compute Softmax with T=2
    s_func = Softmax(temperature=2.0)
    probs_t2 = s_func.forward(logits)

    # 2. Compute expected grad: (p - y) / 2
    expected_grad = (probs_t2 - target) / 2.0

    # Verify
    assert np.allclose(grad2, expected_grad)


def test_cce_gradient_value_check():
    """Verify deterministic gradient value for a specific case.

    Logits: [10.0, 0.0]
    Target: [1.0, 0.0]
    Temperature: 10.0

    1. Scaled Logits: [1.0, 0.0]
    2. Softmax([1.0, 0.0]) = [e/(e+1), 1/(e+1)] ≈ [0.731058, 0.268941]
    3. Diff (p - y): [0.731... - 1, 0.268... - 0] = [-0.268941, 0.268941]
    4. Grad (diff / T): [-0.0268941, 0.0268941]
    """
    logits = np.array([[10.0, 0.0]])
    target = np.array([[1.0, 0.0]])
    T = 10.0

    loss = CategoricalCrossEntropy(temperature=T)
    grad = loss.prime(logits, target)

    # Manual calculation
    exp_logits = np.exp(logits / T)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    expected_grad = (probs - target) / T

    # Check closeness (gradient is per-batch, so prime divides by N=1)
    assert np.allclose(grad, expected_grad)
