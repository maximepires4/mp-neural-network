from abc import abstractmethod

from . import DTYPE, ArrayType, xp
from .activations import Sigmoid, Softmax


class Loss:
    @abstractmethod
    def direct(self, output: ArrayType, output_expected: ArrayType) -> float:
        pass

    @abstractmethod
    def prime(self, output: ArrayType, output_expected: ArrayType) -> ArrayType:
        pass

    def get_config(self) -> dict:
        return {"type": self.__class__.__name__}


class MSE(Loss):
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
    def __init__(self) -> None:
        self.softmax = Softmax()

    def direct(self, output: ArrayType, output_expected: ArrayType) -> float:
        epsilon = 1e-9
        predictions = self.softmax.forward(output)
        res: float = xp.mean(
            -xp.sum(
                output_expected * xp.log(predictions + epsilon, dtype=DTYPE),
                axis=1,
                dtype=DTYPE,
            ),
            dtype=DTYPE,
        )
        return res

    def prime(self, output: ArrayType, output_expected: ArrayType) -> ArrayType:
        predictions = self.softmax.forward(output)
        grad: ArrayType = (predictions - output_expected) / output.shape[0]
        return grad
