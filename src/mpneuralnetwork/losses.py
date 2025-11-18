import numpy as np
from .layers import Layer


class Loss:
    def direct(self, output, output_expected):
        pass

    def prime(self, output, output_expected):
        pass


class MSE(Loss):
    def direct(self, output, output_expected):
        return np.mean(np.power(output_expected - output, 2))

    def prime(self, output, output_expected):
        return 2 * (output_expected - output) / output.size

class BinaryCrossEntropy(Loss):
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def direct(self, output, output_expected):
        return np.mean(np.maximum(output_expected, 0) - output_expected * output + np.log(1 + np.exp(-np.abs(output_expected))))

    def prime(self, output, output_expected):
        return (self._sigmoid(output_expected) - output) / output.size

class CategoricalCrossEntropy(Loss):
    def _softmax(self, x):
        m = np.max(x)
        e = np.exp(x - m)
        self.output = e / np.sum(e)
        return self.output

    def direct(self, output, output_expected):
        epsilon = 1e-9
        predictions = self._softmax(output)
        return -np.sum(output_expected * np.log(predictions + epsilon)) 

    def prime(self, output, output_expected):
        predictions = self._softmax(output)
        return predictions - output_expected
