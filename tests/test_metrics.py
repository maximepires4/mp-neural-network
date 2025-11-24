import numpy as np

from mpneuralnetwork.metrics import (
    MAE,
    RMSE,
    Accuracy,
    F1Score,
    Precision,
    R2Score,
    Recall,
    TopKAccuracy,
)


def test_rmse_perfect():
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[1.0], [2.0], [3.0]])
    metric = RMSE()
    assert metric(y_true, y_pred) == 0.0


def test_rmse_error():
    y_true = np.array([[0.0], [0.0]])
    y_pred = np.array([[3.0], [4.0]])
    # Errors: 3, 4
    # Squared: 9, 16
    # Sum sq per sample: 9, 16
    # Mean: 12.5
    # Sqrt: ~3.535
    metric = RMSE()
    assert np.isclose(metric(y_true, y_pred), np.sqrt(12.5))


def test_mae():
    y_true = np.array([[0.0], [0.0]])
    y_pred = np.array([[3.0], [-4.0]])
    # Abs errors: 3, 4
    # Mean: 3.5
    metric = MAE()
    assert metric(y_true, y_pred) == 3.5


def test_r2_score_perfect():
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[1.0], [2.0], [3.0]])
    metric = R2Score()
    assert metric(y_true, y_pred) == 1.0


def test_r2_score_bad():
    # If we predict the mean, R2 should be 0
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[2.0], [2.0], [2.0]])
    metric = R2Score()
    assert np.isclose(metric(y_true, y_pred), 0.0)


def test_accuracy_binary():
    # Probabilities
    y_true = np.array([[0], [1], [1], [0]])
    y_pred = np.array([[0.1], [0.9], [0.4], [0.2]])
    # Rounded preds: 0, 1, 0, 0
    # Matches: T, T, F, T -> 3/4 = 0.75
    metric = Accuracy()
    assert metric(y_true, y_pred) == 0.75


def test_accuracy_categorical_one_hot():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])
    # Argmax true: 0, 1, 2
    # Argmax pred: 0, 1, 2
    metric = Accuracy()
    assert metric(y_true, y_pred) == 1.0


def test_precision_binary():
    # True: 0, 1, 0, 1
    # Pred: 0, 1, 1, 0
    y_true = np.array([[0], [1], [0], [1]])
    y_pred = np.array([[0], [1], [1], [0]])

    # TP: Pred=1 & True=1 -> Index 1 only. Count = 1.
    # FP: Pred=1 & True=0 -> Index 2 only. Count = 1.
    # Precision = TP / (TP + FP) = 1 / 2 = 0.5
    metric = Precision()
    assert np.isclose(metric(y_true, y_pred), 0.5)


def test_precision_categorical():
    # Multi-class treated as macro/micro depends on implementation.
    # The current implementation uses argmax, essentially treating it as multiclass
    # BUT then it does `y_pred == 1`.
    # Wait, `y_pred` becomes class INDICES (0, 1, 2...).
    # `y_pred == 1` checks if class is 1.
    # This effectively calculates precision for CLASS 1 only (binary precision for class index 1).

    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
    # Indices: 0, 1, 1

    y_pred = np.array([[0.9, 0.1, 0], [0.1, 0.8, 0.1], [0.8, 0.2, 0]])
    # Indices: 0, 1, 0

    # We are checking for CLASS 1 (== 1)
    # True class 1s: Index 1, 2
    # Pred class 1s: Index 1

    # TP (Pred=1 & True=1): Index 1. Count = 1.
    # FP (Pred=1 & True!=1): None. Count = 0.
    # Precision = 1 / 1 = 1.0

    metric = Precision()
    assert np.isclose(metric(y_true, y_pred), 1.0)


def test_recall_binary():
    # True: 0, 1, 0, 1
    # Pred: 0, 1, 1, 0
    y_true = np.array([[0], [1], [0], [1]])
    y_pred = np.array([[0], [1], [1], [0]])

    # TP: 1 (Index 1)
    # FN (Pred=0 & True=1): Index 3. Count = 1.
    # Recall = TP / (TP + FN) = 1 / 2 = 0.5
    metric = Recall()
    assert np.isclose(metric(y_true, y_pred), 0.5)


def test_f1_score():
    # Use same data as Precision/Recall tests
    y_true = np.array([[0], [1], [0], [1]])
    y_pred = np.array([[0], [1], [1], [0]])

    # P = 0.5, R = 0.5
    # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5 / 1.0 = 0.5
    metric = F1Score()
    assert np.isclose(metric(y_true, y_pred), 0.5)


def test_top_k_accuracy():
    # 3 classes. k=2.
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Class indices: 0, 1, 2

    y_pred = np.array(
        [
            [0.2, 0.3, 0.5],  # Top 2: (2, 1). True is 0. FAIL.
            [0.1, 0.4, 0.5],  # Top 2: (2, 1). True is 1. PASS.
            [0.1, 0.4, 0.5],  # Top 2: (2, 1). True is 2. PASS.
        ]
    )

    metric = TopKAccuracy(k=2)
    # 2/3 correct
    assert np.isclose(metric(y_true, y_pred), 2 / 3)
