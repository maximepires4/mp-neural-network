from abc import abstractmethod

from . import DTYPE, ArrayType, xp
from .activations import Sigmoid, Softmax


class Loss:
    """Base class for loss functions.

    Computes the error between the model's predictions and the true values,
    as well as the gradient of this error for backpropagation.
    """

    @abstractmethod
    def direct(self, output: ArrayType, output_expected: ArrayType) -> float:
        """Computes the scalar loss value.

        Args:
            output (ArrayType): Model predictions.
            output_expected (ArrayType): Ground truth values.

        Returns:
            float: The loss value.
        """
        pass

    @abstractmethod
    def prime(self, output: ArrayType, output_expected: ArrayType) -> ArrayType:
        """Computes the gradient of the loss function w.r.t the input.

        Args:
            output (ArrayType): Model predictions.
            output_expected (ArrayType): Ground truth values.

        Returns:
            ArrayType: Gradient of the loss.
        """
        pass

    def get_config(self) -> dict:
        return {"type": self.__class__.__name__}


class MSE(Loss):
    """Mean Squared Error (MSE) Loss.

    Formula:
        `L = (1/N) * sum((y_pred - y_true)^2)`

    Derivative:
        `dL/dy_pred = (2/N) * (y_pred - y_true)`

    Used for regression problems.
    """

    def direct(self, output: ArrayType, output_expected: ArrayType) -> float:
        res: float = xp.mean(
            xp.sum(xp.square(output_expected - output), axis=1, dtype=DTYPE),
            dtype=DTYPE,
        )
        return res

    def prime(self, output: ArrayType, output_expected: ArrayType) -> ArrayType:
        grad: ArrayType = 2 * (output - output_expected) / output.shape[0]
        return grad


class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy Loss.

    Formula (conceptually):
        `L = - (y * log(p) + (1-y) * log(1-p))`

    Implementation details:
        Assumes the input `output` corresponds to **LOGITS** (raw scores), not probabilities.
        The Sigmoid activation is applied internally using the numerically stable log-sum-exp trick.

    Used for binary classification problems.
    """

    def __init__(self) -> None:
        self.sigmoid = Sigmoid()

    def direct(self, output: ArrayType, output_expected: ArrayType) -> float:
        loss_per_element = xp.maximum(output, 0) - output * output_expected + xp.log(1 + xp.exp(-xp.abs(output), dtype=DTYPE), dtype=DTYPE)
        res: float = xp.mean(xp.sum(loss_per_element, axis=1, dtype=DTYPE), dtype=DTYPE)
        return res

    def prime(self, output: ArrayType, output_expected: ArrayType) -> ArrayType:
        predictions = self.sigmoid.forward(output)
        grad: ArrayType = (predictions - output_expected) / output.shape[0]
        return grad


class CategoricalCrossEntropy(Loss):
    """Categorical Cross Entropy Loss.

    Formula (conceptually):
        `L = - sum(y_i * log(p_i))`

    Implementation details:
        Assumes the input `output` corresponds to **LOGITS**.
        The Softmax activation is applied internally.

    Used for multi-class classification problems (one-hot encoded targets).
    """

    def __init__(self, temperature: float = 1.0, epsilon: float = 1e-8) -> None:
        """Initializes the Categorical Cross Entropy loss.

        Args:
            temperature (float, optional): Temperature parameter for the softmax. Defaults to 1.0.
            epsilon (float, optional): Small float added to denominator to avoid dividing by zero. Defaults to 1e-8.
        """
        self.temperature = temperature
        self.epsilon = epsilon
        self.softmax = Softmax(temperature=temperature, epsilon=epsilon)

    def direct(self, output: ArrayType, output_expected: ArrayType) -> float:
        predictions = self.softmax.forward(output)
        res: float = xp.mean(
            -xp.sum(
                output_expected * xp.log(predictions + self.epsilon, dtype=DTYPE),
                axis=1,
                dtype=DTYPE,
            ),
            dtype=DTYPE,
        )
        return res

    def prime(self, output: ArrayType, output_expected: ArrayType) -> ArrayType:
        predictions = self.softmax.forward(output)
        grad: ArrayType = ((predictions - output_expected) / (self.temperature + self.epsilon)) / output.shape[0]
        return grad
