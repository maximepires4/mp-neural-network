from typing import Any

import numpy as np
from numpy.typing import NDArray

from mpneuralnetwork import DTYPE


def check_loss_gradient(loss_fn, y_pred, y_true, epsilon=1e-4, atol=1e-2):
    """
    Helper function to perform numerical gradient checking for a loss function's prime method (dL/dY_pred).

    Args:
        loss_fn: The loss function instance.
        y_pred: Predicted values (logits or probabilities).
        y_true: True target values.
        epsilon: Small value for finite difference.
        atol: Absolute tolerance for comparison.
    """

    # 1. Analytical gradient
    analytical_grad = loss_fn.prime(y_pred.copy(), y_true)

    # 2. Numerical gradient
    numerical_grad = np.zeros_like(y_pred, dtype=DTYPE)

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


def check_gradient(layer: Any, x: NDArray, y: NDArray, loss_fn: Any, epsilon: float = 1e-4, atol: float = 1e-2) -> None:
    """
    Helper function to perform numerical gradient checking for a layer's backward pass (dL/dX).

    Args:
        layer: The layer instance to test.
        x: Input data.
        y: True labels.
        loss_fn: The loss function instance.
        epsilon: A small value for finite difference calculation.
        atol: The absolute tolerance for comparing analytical and numerical gradients.
    """
    # 1. Calculate analytical gradient (the one computed by the backward method)
    preds = layer.forward(x.copy())
    output_gradient = loss_fn.prime(preds, y)
    analytical_grads_x = layer.backward(output_gradient)

    # 2. Calculate numerical gradient (the "true" gradient using finite differences)
    numerical_grads_x = np.zeros_like(x, dtype=DTYPE)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        # Save original value
        original_value = x[ix]

        # Calculate loss for x + epsilon
        x[ix] = original_value + epsilon
        preds_plus = layer.forward(x.copy())
        # Summing loss over batch and features to get a scalar for the derivative
        loss_plus = np.sum(loss_fn.direct(preds_plus, y))

        # Calculate loss for x - epsilon
        x[ix] = original_value - epsilon
        preds_minus = layer.forward(x.copy())
        loss_minus = np.sum(loss_fn.direct(preds_minus, y))

        # Restore original value
        x[ix] = original_value

        # Compute the slope and store it
        numerical_grads_x[ix] = (loss_plus - loss_minus) / (2 * epsilon)

        it.iternext()

    # 3. Assert that the two gradients are close
    assert np.allclose(analytical_grads_x, numerical_grads_x, atol=atol), f"Gradient mismatch for layer {layer.__class__.__name__}"
