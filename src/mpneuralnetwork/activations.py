import numpy as np
from .layers import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_batch, training=True):
        self.input = input_batch
        return self.activation(self.input)

    def backward(self, output_gradient_batch):
        return np.multiply(
            output_gradient_batch, self.activation_prime(self.input)
        )


class Tanh(Activation):
    def __init__(self):
        super().__init__(np.tanh, lambda x: 1 - np.tanh(x) ** 2)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        super().__init__(lambda x: sigmoid(x), lambda x: sigmoid(x) * (1 - sigmoid(x)))


class ReLU(Activation):
    def __init__(self):
        super().__init__(lambda x: np.maximum(0, x), lambda x: x > 0)


class PReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__(
            lambda x: np.maximum(alpha * x, x), lambda x: np.where(x < 0, alpha, 1)
        )


class Swish(Activation):
    def __init__(self):
        super().__init__(
            lambda x: x / (1 + np.exp(-x)),
            lambda x: (1 + np.exp(-x) + x * np.exp(-x)) / (1 + np.exp(-x)) ** 2,
        )


class Softmax(Layer):
    def __init__(self):
        pass

    def forward(self, input_batch, training=True):
        m = np.max(input_batch, axis=1, keepdims=True)
        e = np.exp(input_batch - m)
        self.output = e / np.sum(e, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient_batch):
        sum_s_times_g = np.sum(
            self.output * output_gradient_batch, axis=1, keepdims=True
        )

        return self.output * (output_gradient_batch - sum_s_times_g)
