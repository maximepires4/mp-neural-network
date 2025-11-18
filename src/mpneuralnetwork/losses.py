import numpy as np
from .layers import Layer


class Loss:
    def direct(self, output, output_expected):
        pass

    def prime(self, output, output_expected):
        pass


class MSE(Loss):
    def direct(self, output, output_expected):
        return np.mean(np.sum(np.power(output_expected - output, 2), axis=1))

    def prime(self, output, output_expected):
        return 2 * (output - output_expected) / output.shape[0]


class BinaryCrossEntropy(Loss):
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def direct(self, output, output_expected):
        loss_per_element = (
            np.maximum(output, 0)
            - output * output_expected
            + np.log(1 + np.exp(-np.abs(output)))
        )
        return np.mean(np.sum(loss_per_element, axis=1))

    def prime(self, output, output_expected):
        predictions = self._sigmoid(output)
        return (predictions - output_expected) / output.shape[0]


class CategoricalCrossEntropy(Loss):
    def _softmax(self, x):
        m = np.max(x, axis=1, keepdims=True)
        e = np.exp(x - m)
        self.output = e / np.sum(e, axis=1, keepdims=True)
        return self.output

    def direct(self, output, output_expected):
        epsilon = 1e-9
        predictions = self._softmax(output)
        return np.mean(-np.sum(output_expected * np.log(predictions + epsilon), axis=1))

    def prime(self, output, output_expected):
        predictions = self._softmax(output)
        return (predictions - output_expected) / output.shape[0]
