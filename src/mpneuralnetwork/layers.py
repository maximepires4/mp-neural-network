import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_gradient):
        pass

    def clear_gradients(self):
        pass

    def average_gradients(self, batch_size):
        pass

    def update(self, learning_rate, batch_size):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
        self.weights_gradient = np.zeros((output_size, input_size))
        self.output_gradient = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        output = self.weights @ self.input + self.biases
        return output

    def backward(self, output_gradient):
        self.weights_gradient += output_gradient @ self.input.T
        self.output_gradient += output_gradient

        return self.weights.T @ output_gradient  # input_gradient

    def clear_gradients(self):
        self.weights_gradient[:] = 0
        self.output_gradient[:] = 0

    def update(self, learning_rate, batch_size):
        self.weights_gradient /= batch_size
        self.output_gradient /= batch_size

        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * self.output_gradient
