import numpy as np
from scipy import signal


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


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (
            depth,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        )
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        self.kernels_gradient = np.zeros(self.kernels_shape)
        self.output_gradient = np.zeros(self.output_shape)

    def forward(self, input):
        self.input = input
        output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i][j], "valid"
                )
        return output

    def backward(self, output_gradient):
        self.output_gradient += output_gradient
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.kernels_gradient[i][j] += signal.correlate2d(
                    self.input[j], output_gradient[i], "valid"
                )
                input_gradient[j] += signal.convolve2d(
                    output_gradient[i], self.kernels[i][j], "full"
                )

        return input_gradient

    def clear_gradients(self):
        self.kernels_gradient[:] = 0
        self.output_gradient[:] = 0

    def update(self, learning_rate, batch_size):
        self.kernels_gradient /= batch_size
        self.output_gradient /= batch_size

        self.kernels -= learning_rate * self.kernels_gradient
        self.biases -= learning_rate * self.output_gradient


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient):
        return np.reshape(output_gradient, self.input_shape)
