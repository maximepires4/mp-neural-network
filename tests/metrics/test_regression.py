import numpy as np

from mpneuralnetwork.metrics import MAE, RMSE, R2Score


def test_rmse_perfect():
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[1.0], [2.0], [3.0]])
    metric = RMSE()
    assert metric(y_true, y_pred) == 0.0


def test_rmse_error():
    y_true = np.array([[0.0], [0.0]])
    y_pred = np.array([[3.0], [4.0]])
    metric = RMSE()
    assert np.isclose(metric(y_true, y_pred), np.sqrt(12.5))


def test_mae():
    y_true = np.array([[0.0], [0.0]])
    y_pred = np.array([[3.0], [-4.0]])
    metric = MAE()
    assert metric(y_true, y_pred) == 3.5


def test_r2_score_perfect():
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[1.0], [2.0], [3.0]])
    metric = R2Score()
    assert metric(y_true, y_pred) == 1.0


def test_r2_score_bad():
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[2.0], [2.0], [2.0]])
    metric = R2Score()
    assert np.isclose(metric(y_true, y_pred), 0.0)
