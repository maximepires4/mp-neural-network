from abc import abstractmethod

from mpneuralnetwork import DTYPE, ArrayType, xp


class Metric:
    def get_config(self) -> dict:
        return {"type": self.__class__.__name__}

    @abstractmethod
    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        pass


class RMSE(Metric):
    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        mse = xp.mean(xp.sum(xp.square(y_true - y_pred), axis=1, dtype=DTYPE), dtype=DTYPE)
        return self.from_mse(float(mse))

    def from_mse(self, mse: float) -> float:
        res: float = xp.sqrt(mse, dtype=DTYPE)
        return res


class MAE(Metric):
    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        res: float = xp.mean(xp.sum(xp.abs(y_true - y_pred), axis=1, dtype=DTYPE), dtype=DTYPE)
        return res


class R2Score(Metric):
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
    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true = xp.argmax(y_true, axis=1)
            y_pred = xp.argmax(y_pred, axis=1)
        else:
            y_pred = xp.round(y_pred)

        res: float = xp.mean(y_true == y_pred, dtype=DTYPE)
        return res


class Precision(Metric):
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
