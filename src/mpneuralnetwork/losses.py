from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray


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
        return float(np.mean(np.sum(np.power(output_expected - output, 2), axis=1)))

    def prime(self, output: NDArray, output_expected: NDArray) -> NDArray:
        grad: NDArray = 2 * (output - output_expected) / output.shape[0]
        return grad


class BinaryCrossEntropy(Loss):
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def direct(self, output: NDArray, output_expected: NDArray) -> float:
        loss_per_element = np.maximum(output, 0) - output * output_expected + np.log(1 + np.exp(-np.abs(output)))
        return float(np.mean(np.sum(loss_per_element, axis=1)))

    def prime(self, output: NDArray, output_expected: NDArray) -> NDArray:
        predictions = self._sigmoid(output)
        grad: NDArray = (predictions - output_expected) / output.shape[0]
        return grad


class CategoricalCrossEntropy(Loss):
    def _softmax(self, x):
        m = np.max(x, axis=1, keepdims=True)
        e = np.exp(x - m)
        self.output = e / np.sum(e, axis=1, keepdims=True)
        return self.output

    def direct(self, output: NDArray, output_expected: NDArray) -> float:
        epsilon = 1e-9
        predictions = self._softmax(output)
        return float(np.mean(-np.sum(output_expected * np.log(predictions + epsilon), axis=1)))

    def prime(self, output: NDArray, output_expected: NDArray) -> NDArray:
        predictions = self._softmax(output)
        grad: NDArray = (predictions - output_expected) / output.shape[0]
        return grad
