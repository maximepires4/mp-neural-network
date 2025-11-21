from abc import abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal

Lit_W = Literal["auto", "he", "xavier"]


class Layer:
    def __init__(self) -> None:
        self.input: NDArray
        self.output: NDArray
        self.input_size: int
        self.output_size: int

    def build(self, input_size: int) -> None:
        self.input_size = input_size
        self.output_size = input_size

    @abstractmethod
    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        pass

    @abstractmethod
    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        pass

    @property
    def params(self) -> dict[str, tuple[NDArray, NDArray]]:
        return {}


class Dense(Layer):
    def __init__(self, output_size: int, input_size: int | None = None, initialization: Lit_W = "auto") -> None:
        self.input_size: int
        self.output_size: int = output_size
        self.initialization: Lit_W = initialization

        self.weights: NDArray
        self.weights_gradient: NDArray

        self.biases: NDArray
        self.biases_gradient: NDArray

        if input_size is not None:
            self.build(input_size)

    def build(self, input_size: int):
        self.input_size = input_size

        self.biases = np.random.randn(1, self.output_size)
        self.biases_gradient = np.zeros_like(self.biases)

        if self.initialization != "auto":
            self.init_weights(self.initialization)

    def init_weights(self, method: Lit_W) -> None:
        std_dev = 0.1

        if method == "he":
            std_dev = np.sqrt(2.0 / self.input_size)
        elif method == "xavier":
            std_dev = np.sqrt(1.0 / self.input_size)

        self.weights = np.random.randn(self.input_size, self.output_size) * std_dev
        self.weights_gradient = np.zeros_like(self.weights)

    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        self.input = input_batch
        res: NDArray = self.input @ self.weights + self.biases
        return res

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        self.weights_gradient = self.input.T @ output_gradient_batch
        self.biases_gradient = np.sum(output_gradient_batch, axis=0, keepdims=True)

        grad: NDArray = output_gradient_batch @ self.weights.T
        return grad

    @property
    def params(self) -> dict[str, tuple[NDArray, NDArray]]:
        return {
            "weights": (self.weights, self.weights_gradient),
            "biases": (self.biases, self.biases_gradient),
        }


class Dropout(Layer):
    def __init__(self, probability: float = 0.5) -> None:
        self.probability: float = probability
        self.mask: NDArray

    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        if not training:
            return input_batch

        self.mask = np.random.binomial(1, 1 - self.probability, size=input_batch.shape) / (1 - self.probability)

        res: NDArray = input_batch * self.mask
        return res

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        grad: NDArray = output_gradient_batch * self.mask
        return grad


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

    def forward(self, input_batch, training=True):
        # TODO: Need to vectorize this part. For now, only works for batch_size = 1
        assert input_batch.ndim == 3, f"Non-vectorized Convolutional layer received a batch of size > 1 (shape={input_batch.shape})"

        self.input = input_batch
        output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                output[i] += signal.correlate2d(self.input[j], self.kernels[i][j], "valid")
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
                self.kernels_gradient[i][j] = signal.correlate2d(self.input[j], output_gradient_batch[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient_batch[i], self.kernels[i][j], "full")

        return input_gradient

    @property
    def params(self) -> dict[str, tuple[NDArray, NDArray]]:
        # TODO: Add this
        return {}


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input_batch, training=True):
        batch_size = input_batch.shape[-1]
        return np.reshape(input_batch, (*self.output_shape, batch_size))

    def backward(self, output_gradient_batch):
        batch_size = output_gradient_batch.shape[-1]
        return np.reshape(output_gradient_batch, (*self.input_shape, batch_size))

    @property
    def params(self) -> dict[str, tuple[NDArray, NDArray]]:
        # TODO: Add this
        return {}
