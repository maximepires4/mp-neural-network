from abc import abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal

Lit_W = Literal["auto", "he", "xavier"]


class Layer:
    def __init__(self) -> None:
        # TODO: Call super for setting `input_size` so that every layer can be a first layer
        self.input: NDArray
        self.output: NDArray
        self.input_size: int
        self.output_size: int

    def get_config(self) -> dict:
        return {"type": self.__class__.__name__}

    def build(self, input_size: int) -> None:
        self.input_size = input_size

        if not hasattr(self, "output_size"):
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

    def load_params(self, params: dict[str, NDArray]) -> None:
        pass

    @property
    def state(self) -> dict[str, NDArray]:
        return {}

    @state.setter
    def state(self, state: dict[str, NDArray]) -> None:
        pass


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

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"output_size": self.output_size, "input_size": self.input_size, "initialization": self.initialization})
        return config

    def build(self, input_size: int):
        super().build(input_size)

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

    def load_params(self, params: dict[str, NDArray]) -> None:
        self.weights = params["weights"]
        self.biases = params["biases"]


class Dropout(Layer):
    def __init__(self, probability: float = 0.5) -> None:
        self.probability: float = probability
        self.mask: NDArray

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"probability": self.probability})
        return config

    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        if not training:
            return input_batch

        self.mask = np.random.binomial(1, 1 - self.probability, size=input_batch.shape) / (1 - self.probability)

        res: NDArray = input_batch * self.mask
        return res

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        grad: NDArray = output_gradient_batch * self.mask
        return grad


class BatchNormalization(Layer):
    def __init__(self, momentum: float = 0.9, epsilon: float = 1e-8) -> None:
        self.momentum: float = momentum
        self.epsilon: float = epsilon

        self.gamma: NDArray
        self.beta: NDArray

        self.cache_m: NDArray
        self.cache_v: NDArray

    def build(self, input_size: int) -> None:
        super().build(input_size)

        self.gamma = np.ones((1, self.input_size))
        self.gamma_gradient = np.zeros_like(self.gamma)

        self.beta = np.zeros((1, self.input_size))
        self.beta_gradient = np.zeros_like(self.beta)

        self.cache_m = np.zeros((1, input_size))
        self.cache_v = np.ones((1, input_size))

        self.std_inv: NDArray
        self.x_centered: NDArray
        self.x_norm: NDArray

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"momentum": self.momentum, "epsilon": self.epsilon})
        return config

    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        self.input = input_batch

        mean: NDArray
        var: NDArray

        if training:
            mean = np.mean(self.input, axis=0, keepdims=True)
            var = np.var(self.input, axis=0, keepdims=True)

            self.cache_m = self.momentum * self.cache_m + (1 - self.momentum) * mean
            self.cache_v = self.momentum * self.cache_v + (1 - self.momentum) * var

        else:
            mean = self.cache_m
            var = self.cache_v

        self.std_inv = 1 / np.sqrt(var + self.epsilon)
        self.x_centered = self.input - mean
        self.x_norm = self.x_centered * self.std_inv

        res: NDArray = self.x_norm * self.gamma + self.beta
        return res

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        self.gamma_gradient = np.sum(self.x_norm * output_gradient_batch, axis=0, keepdims=True)
        self.beta_gradient = np.sum(output_gradient_batch, axis=0, keepdims=True)

        N = output_gradient_batch.shape[0]
        dx_norm = output_gradient_batch * self.gamma

        grad: NDArray = (
            (1 / N)
            * self.std_inv
            * (N * dx_norm - np.sum(dx_norm, axis=0, keepdims=True) - self.x_norm * np.sum(dx_norm * self.x_norm, axis=0, keepdims=True))
        )
        return grad

    @property
    def params(self) -> dict[str, tuple[NDArray, NDArray]]:
        return {"gamma": (self.gamma, self.gamma_gradient), "beta": (self.beta, self.beta_gradient)}

    def load_params(self, params: dict[str, NDArray]) -> None:
        self.gamma = params["gamma"]
        self.beta = params["beta"]

    @property
    def state(self) -> dict[str, NDArray]:
        return {"cache_m": self.cache_m, "cache_v": self.cache_v}

    @state.setter
    def state(self, state: dict[str, NDArray]) -> None:
        self.cache_m = state["cache_m"]
        self.cache_v = state["cache_v"]


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
