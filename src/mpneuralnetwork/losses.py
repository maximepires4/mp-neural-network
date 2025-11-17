import numpy as np
from .layers import Layer


class Loss:
    def __init__(self):
        self.output = None
        self.output_expected = None

    def direct(self, output, output_expected):
        pass

    def prime(self, output, output_expected):
        pass


class MSE(Loss):
    def direct(self, output, output_expected):
        return np.mean(np.power(output_expected - output, 2))

    def prime(self, output, output_expected):
        return 2 * (output_expected - output) / output.size

class CrossEntropy(Loss):
    def direct(self, output, output_expected):
        epsilon = 1e-9
        return -np.sum(output_expected * np.log(output + epsilon))

    def prime(self, output, output_expected):
        return output_expected - output
