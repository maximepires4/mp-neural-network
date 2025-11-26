from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from . import DTYPE
from .activations import Sigmoid, Softmax


class Loss:
    @abstractmethod
    def direct(self, output: NDArray, output_expected: NDArray) -> float:
        pass

    @abstractmethod
    def prime(self, output: NDArray, output_expected: NDArray) -> NDArray:
        pass

    def get_config(self) -> dict:
        return {"type": self.__class__.__name__}


class MSE(Loss):
    def direct(self, output: NDArray, output_expected: NDArray) -> float:
        return float(
            np.mean(
                np.sum(np.square(output_expected - output), axis=1, dtype=DTYPE),
                dtype=DTYPE,
            )
        )

    def prime(self, output: NDArray, output_expected: NDArray) -> NDArray:
        grad: NDArray = 2 * (output - output_expected) / output.shape[0]
        return grad


class BinaryCrossEntropy(Loss):
    def __init__(self) -> None:
        self.sigmoid = Sigmoid()

    def direct(self, output: NDArray, output_expected: NDArray) -> float:
        loss_per_element = np.maximum(output, 0) - output * output_expected + np.log(1 + np.exp(-np.abs(output), dtype=DTYPE), dtype=DTYPE)
        return float(np.mean(np.sum(loss_per_element, axis=1, dtype=DTYPE), dtype=DTYPE))

    def prime(self, output: NDArray, output_expected: NDArray) -> NDArray:
        predictions = self.sigmoid.forward(output)
        grad: NDArray = (predictions - output_expected) / output.shape[0]
        return grad


class CategoricalCrossEntropy(Loss):
    def __init__(self) -> None:
        self.softmax = Softmax()

    def direct(self, output: NDArray, output_expected: NDArray) -> float:
        epsilon = 1e-9
        predictions = self.softmax.forward(output)
        return float(
            np.mean(
                -np.sum(
                    output_expected * np.log(predictions + epsilon, dtype=DTYPE),
                    axis=1,
                    dtype=DTYPE,
                ),
                dtype=DTYPE,
            )
        )

    def prime(self, output: NDArray, output_expected: NDArray) -> NDArray:
        predictions = self.softmax.forward(output)
        grad: NDArray = (predictions - output_expected) / output.shape[0]
        return grad
