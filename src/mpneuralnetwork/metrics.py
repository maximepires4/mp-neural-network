from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray


class Metric:
    def get_config(self) -> dict:
        return {"type": self.__class__.__name__}

    @abstractmethod
    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        pass


class RMSE(Metric):
    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        return self.from_mse(np.mean(np.sum(np.square(y_true - y_pred), axis=1)))

    def from_mse(self, mse: float) -> float:
        return float(np.sqrt(mse))


class MAE(Metric):
    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        return float(np.mean(np.sum(np.abs(y_true - y_pred), axis=1)))


class R2Score(Metric):
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon: float = epsilon

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        var_tp = np.sum(np.square(y_true - y_pred))
        var_tm = np.sum(np.square(y_true - np.mean(y_true, axis=0)))

        return float(1 - var_tp / (var_tm + self.epsilon))


class Accuracy(Metric):
    def __call__(self, y_true: NDArray, y_pred: NDArray, no_check: bool = False) -> float:
        if not no_check:
            if y_true.ndim == 2 and y_true.shape[1] > 1:
                y_true = np.argmax(y_true, axis=1)
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = np.round(y_pred)

        return float(np.mean(y_true == y_pred))


class Precision(Metric):
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon: float = epsilon

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    def __call__(self, y_true: NDArray, y_pred: NDArray, no_check: bool = False) -> float:
        if not no_check:
            if y_true.ndim == 2 and y_true.shape[1] > 1:
                y_true = np.argmax(y_true, axis=1)
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = np.round(y_pred)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        return float(tp / (tp + fp + self.epsilon))


class Recall(Metric):
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon: float = epsilon

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    def __call__(self, y_true: NDArray, y_pred: NDArray, no_check: bool = False) -> float:
        if not no_check:
            if y_true.ndim == 2 and y_true.shape[1] > 1:
                y_true = np.argmax(y_true, axis=1)
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = np.round(y_pred)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        return float(tp / (tp + fn + self.epsilon))


class F1Score(Metric):
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon: float = epsilon

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = np.round(y_pred)

        precision = Precision(self.epsilon)(y_true, y_pred, no_check=True)
        recall = Recall(self.epsilon)(y_true, y_pred, no_check=True)

        return 2 * precision * recall / (precision + recall + self.epsilon)


class TopKAccuracy(Metric):
    def __init__(self, k: int) -> None:
        self.k: int = k

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"k": self.k})
        return config

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        top_k_preds = np.argsort(y_pred, axis=1)[:, -self.k :]

        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        y_true = y_true.reshape(-1, 1)

        return float(np.mean(np.any(top_k_preds == y_true, axis=1)))
