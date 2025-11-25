import numpy as np

from mpneuralnetwork.metrics import Accuracy, F1Score, Precision, Recall, TopKAccuracy


def test_accuracy_binary():
    y_true = np.array([[0], [1], [1], [0]])
    y_pred = np.array([[0.1], [0.9], [0.4], [0.2]])
    metric = Accuracy()
    assert metric(y_true, y_pred) == 0.75


def test_accuracy_categorical_one_hot():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])
    metric = Accuracy()
    assert metric(y_true, y_pred) == 1.0


def test_precision_binary():
    y_true = np.array([[0], [1], [0], [1]])
    y_pred = np.array([[0], [1], [1], [0]])
    metric = Precision()
    assert np.isclose(metric(y_true, y_pred), 0.5)


def test_precision_categorical():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
    y_pred = np.array([[0.9, 0.1, 0], [0.1, 0.8, 0.1], [0.8, 0.2, 0]])
    metric = Precision()
    assert np.isclose(metric(y_true, y_pred), 0.5)


def test_recall_binary():
    y_true = np.array([[0], [1], [0], [1]])
    y_pred = np.array([[0], [1], [1], [0]])
    metric = Recall()
    assert np.isclose(metric(y_true, y_pred), 0.5)


def test_f1_score():
    y_true = np.array([[0], [1], [0], [1]])
    y_pred = np.array([[0], [1], [1], [0]])
    metric = F1Score()
    assert np.isclose(metric(y_true, y_pred), 0.5)


def test_top_k_accuracy():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
            [0.1, 0.4, 0.5],
        ]
    )
    metric = TopKAccuracy(k=2)
    assert np.isclose(metric(y_true, y_pred), 2 / 3)
