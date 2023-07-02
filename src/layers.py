import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
        self.weights_gradient = np.zeros((output_size, input_size))
        self.output_gradient = np.zeros((output_size))

    def forward(self, input):
        self.input = input
        output = self.weights @ self.input + self.biases
        return output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = output_gradient @ self.input.T
        input_gradient = self.weights.T @ output_gradient
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
    
    def update(self, weights, biases):
        self.weights = weights
        self.biases = biases
        return self
