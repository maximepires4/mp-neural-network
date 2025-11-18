import numpy as np
from scipy import signal


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_batch):
        pass

    def backward(self, output_gradient_batch):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(1, output_size)

        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)

    def forward(self, input_batch):
        self.input = input_batch
        return self.input @ self.weights + self.biases

    def backward(self, output_gradient_batch):
        self.weights_gradient = self.input.T @ output_gradient_batch
        self.biases_gradient = np.sum(output_gradient_batch, axis=0, keepdims=True)

        return output_gradient_batch @ self.weights.T


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
        self.biases_gradient = np.zeros(self.output_shape)

    def forward(self, input_batch):
        # TODO: Need to vectorize this part. For now, only works for batch_size = 1
        assert input_batch.ndim == 3, (
            f"Non-vectorized Convolutional layer received a batch of size > 1 (shape={input_batch.shape})"
        )

        self.input = input_batch
        output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i][j], "valid"
                )
        return output

    def backward(self, output_gradient_batch):
        # TODO: Need to vectorize this part. For now, only works for batch_size = 1
        assert output_gradient_batch.ndim == 3, (
            f"Non-vectorized Convolutional layer received a batch of size > 1 (shape={output_gradient_batch.shape})"
        )

        self.kernels_gradient = np.zeros(self.kernels_shape)
        self.biases_gradient = output_gradient_batch

        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.kernels_gradient[i][j] = signal.correlate2d(
                    self.input[j], output_gradient_batch[i], "valid"
                )
                input_gradient[j] += signal.convolve2d(
                    output_gradient_batch[i], self.kernels[i][j], "full"
                )

        return input_gradient


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input_batch):
        batch_size = input_batch.shape[-1]
        return np.reshape(input_batch, (*self.output_shape, batch_size))

    def backward(self, output_gradient_batch):
        batch_size = output_gradient_batch.shape[-1]
        return np.reshape(output_gradient_batch, (*self.input_shape, batch_size))
