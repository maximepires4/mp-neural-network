import numpy as np
import pytest

from mpneuralnetwork.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy, Loss
from tests.utils import check_loss_gradient


def test_loss_configs():
    class MockLoss(Loss):
        def direct(self, o, e):
            pass

        def prime(self, o, e):
            pass

    loss = MockLoss()
    assert loss.get_config() == {"type": "MockLoss"}
    mse = MSE()
    assert mse.get_config() == {"type": "MSE"}


@pytest.mark.parametrize(
    "loss_class, y_pred_shape, y_true_shape",
    [
        (MSE, (64, 10), (64, 10)),
        (BinaryCrossEntropy, (32, 1), (32, 1)),
        (CategoricalCrossEntropy, (16, 5), (16, 5)),
    ],
)
def test_loss_gradient_shape(loss_class, y_pred_shape, y_true_shape):
    loss_fn = loss_class()
    y_pred = np.random.randn(*y_pred_shape)
    y_true = np.random.randn(*y_true_shape)
    gradient = loss_fn.prime(y_pred, y_true)
    assert gradient.shape == y_pred.shape


@pytest.mark.parametrize(
    "loss_class, y_pred_shape, y_true_shape",
    [
        (MSE, (4, 5), (4, 5)),
        (BinaryCrossEntropy, (8, 1), (8, 1)),
        (CategoricalCrossEntropy, (16, 10), (16, 10)),
    ],
)
def test_loss_numerical_gradients(loss_class, y_pred_shape, y_true_shape):
    np.random.seed(69)
    loss_fn = loss_class()
    y_pred = np.random.randn(*y_pred_shape)
    y_true = np.random.randn(*y_true_shape)

    if isinstance(loss_fn, CategoricalCrossEntropy):
        y_true = np.eye(y_true_shape[1])[np.random.choice(y_true_shape[1], y_true_shape[0])]
    if isinstance(loss_fn, BinaryCrossEntropy):
        y_true = np.random.randint(0, 2, size=y_true_shape)

    check_loss_gradient(loss_fn, y_pred, y_true)
