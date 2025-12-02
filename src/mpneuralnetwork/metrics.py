from abc import abstractmethod

from mpneuralnetwork import DTYPE, ArrayType, xp


class Metric:
    """Base class for evaluation metrics.

    Metrics are used to judge the performance of the model. Unlike Loss functions,
    metrics are not used during backpropagation (optimization), only for reporting.
    """

    def get_config(self) -> dict:
        """Returns the metric configuration."""
        return {"type": self.__class__.__name__}

    @abstractmethod
    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        """Computes the metric value.

        Args:
            y_true (ArrayType): Ground truth values.
            y_pred (ArrayType): Model predictions (probabilities or values).

        Returns:
            float: The metric score.
        """
        pass


class RMSE(Metric):
    """Root Mean Squared Error.

    Formula:
        `RMSE = sqrt( (1/N) * sum((y_pred - y_true)^2) )`

    Used primarily for regression tasks. Lower is better.
    """

    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        mse = xp.mean(xp.sum(xp.square(y_true - y_pred), axis=1, dtype=DTYPE), dtype=DTYPE)
        return self.from_mse(float(mse))

    def from_mse(self, mse: float) -> float:
        """Helper to compute RMSE from an existing MSE value."""
        res: float = xp.sqrt(mse, dtype=DTYPE)
        return res


class MAE(Metric):
    """Mean Absolute Error.

    Formula:
        `MAE = (1/N) * sum( |y_pred - y_true| )`

    Used for regression. Less sensitive to outliers than RMSE. Lower is better.
    """

    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        res: float = xp.mean(xp.sum(xp.abs(y_true - y_pred), axis=1, dtype=DTYPE), dtype=DTYPE)
        return res


class R2Score(Metric):
    """R^2 Score (Coefficient of Determination).

    Measures how well the regression predictions approximate the real data points.

    Formula:
        `R2 = 1 - (SS_res / SS_tot)`
        `SS_res = sum((y_true - y_pred)^2)`
        `SS_tot = sum((y_true - mean(y_true))^2)`

    Range: (-inf, 1.0].
    1.0 is perfect prediction. 0.0 is equivalent to a constant model predicting the mean.
    Negative values indicate the model is worse than just predicting the mean.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon: float = epsilon

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        var_tp = xp.sum(xp.square(y_true - y_pred), dtype=DTYPE)
        var_tm = xp.sum(xp.square(y_true - xp.mean(y_true, axis=0, dtype=DTYPE)), dtype=DTYPE)

        res: float = 1 - var_tp / (var_tm + self.epsilon)
        return res


class Accuracy(Metric):
    """Classification Accuracy.

    Formula:
        `Accuracy = (TP + TN) / Total Samples`

    Works for:
    - Binary classification (threshold at 0.5).
    - Multi-class classification (argmax).
    """

    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true = xp.argmax(y_true, axis=1)
            y_pred = xp.argmax(y_pred, axis=1)
        else:
            y_pred = xp.round(y_pred)

        res: float = xp.mean(y_true == y_pred, dtype=DTYPE)
        return res


class Precision(Metric):
    """Precision Metric (Positive Predictive Value).

    Formula:
        `Precision = TP / (TP + FP)`

    Measures the proportion of positive identifications that were actually correct.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon: float = epsilon

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    def __call__(
        self,
        y_true: ArrayType,
        y_pred: ArrayType,
        num_classes: int = 1,
        no_check: bool = False,
    ) -> float:
        if not no_check:
            num_classes = y_true.shape[1]
            if y_true.ndim == 2 and num_classes > 1:
                y_true = xp.argmax(y_true, axis=1)
                y_pred = xp.argmax(y_pred, axis=1)
            else:
                y_pred = xp.round(y_pred)

        sum_score: float = 0
        for c in range(num_classes):
            tp = xp.sum((y_pred == c) & (y_true == c))
            fp = xp.sum((y_pred == c) & (y_true != c))

            sum_score += tp / (tp + fp + self.epsilon)

        return sum_score / num_classes


class Recall(Metric):
    """Recall Metric (Sensitivity / True Positive Rate).

    Formula:
        `Recall = TP / (TP + FN)`

    Measures the proportion of actual positives that were identified correctly.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon: float = epsilon

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    def __call__(
        self,
        y_true: ArrayType,
        y_pred: ArrayType,
        num_classes: int = 1,
        no_check: bool = False,
    ) -> float:
        if not no_check:
            num_classes = y_true.shape[1]
            if y_true.ndim == 2 and num_classes > 1:
                y_true = xp.argmax(y_true, axis=1)
                y_pred = xp.argmax(y_pred, axis=1)
            else:
                y_pred = xp.round(y_pred)

        sum_score: float = 0
        for c in range(num_classes):
            tp = xp.sum((y_pred == c) & (y_true == c))
            fn = xp.sum((y_pred != c) & (y_true == c))
            sum_score += tp / (tp + fn + self.epsilon)

        return sum_score / num_classes


class F1Score(Metric):
    """F1 Score.

    Formula:
        `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

    Harmonic mean of Precision and Recall. Useful for imbalanced datasets.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon: float = epsilon

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        num_classes = y_true.shape[1]
        if y_true.ndim == 2 and num_classes > 1:
            y_true = xp.argmax(y_true, axis=1)
            y_pred = xp.argmax(y_pred, axis=1)
        else:
            y_pred = xp.round(y_pred)

        precision = Precision(self.epsilon)(y_true, y_pred, num_classes=num_classes, no_check=True)
        recall = Recall(self.epsilon)(y_true, y_pred, num_classes=num_classes, no_check=True)

        return 2 * precision * recall / (precision + recall + self.epsilon)


class TopKAccuracy(Metric):
    """Top-K Accuracy.

    Consider the prediction correct if the true label is among the top K probabilities.
    Commonly used in ImageNet classification (Top-5).

    Args:
        k (int): Number of top predictions to consider.
    """

    def __init__(self, k: int) -> None:
        self.k: int = k

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"k": self.k})
        return config

    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        # TODO: no_check ?
        top_k_preds = xp.argsort(y_pred, axis=1)[:, -self.k :]

        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true = xp.argmax(y_true, axis=1)

        y_true = y_true.reshape(-1, 1)

        res: float = xp.mean(xp.any(top_k_preds == y_true, axis=1), dtype=DTYPE)
        return res
