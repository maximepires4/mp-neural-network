import numpy as np
from .layers import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Tanh(Activation):
    def __init__(self):
        super().__init__(np.tanh, lambda x: 1 - np.tanh(x) ** 2)


class Sigmoid(Activation):
    def __init__(self):

        sigmoid = (lambda x: 1 / (1 + np.exp(-x)))

        super().__init__(sigmoid, sigmoid * (1 - sigmoid))


class ReLU(Activation):
    def __init__(self):
        super().__init__(lambda x: np.maximum(0, x), lambda x: x > 0)


class PReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__(lambda x: np.maximum(alpha * x, x), lambda x: alpha if x < 0 else 1)


class Swish(Activation):
    def __init__(self):
        super().__init__(lambda x: x / (1 + np.exp(-x)),
                         lambda x: (1 + np.exp(-x) + x * np.exp(-x)) / (1 + np.exp(-x)) ** 2)


class Softmax(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        m = np.max(input)
        e = np.exp(input - m)
        self.output = e / np.sum(e)
        return self.output

    def backward(self, output_gradient):
        n = np.size(self.output)
        return ((np.identity(n) - self.output.T) * self.output) @ output_gradient
