from abc import abstractmethod
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

Lit_W = Literal["auto", "he", "xavier"]


class Layer:
    def __init__(self, output_shape: int | tuple[int, ...] | None = None, input_shape: int | tuple[int, ...] | None = None) -> None:
        self.output_shape: tuple[int, ...]
        if output_shape is not None:
            if isinstance(output_shape, int):
                output_shape = (output_shape,)
            self.output_shape = output_shape

        self.input_shape: tuple[int, ...]
        if input_shape is not None:
            if isinstance(input_shape, int):
                input_shape = (input_shape,)
            self.input_shape = input_shape

        self.input: NDArray
        self.output: NDArray

    def get_config(self) -> dict:
        return {"type": self.__class__.__name__}

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        self.input_shape = input_shape

        if not hasattr(self, "output_shape"):
            self.output_shape = input_shape

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

    @property
    def input_size(self) -> int:
        return int(np.prod(self.input_shape))

    @property
    def output_size(self) -> int:
        return int(np.prod(self.output_shape))


class Dense(Layer):
    def __init__(self, output_size: int, input_size: int | None = None, initialization: Lit_W = "auto", no_bias: bool = False) -> None:
        super().__init__(output_shape=output_size, input_shape=input_size)
        self.initialization: Lit_W = initialization
        self.no_bias: bool = no_bias

        self.weights: NDArray
        self.weights_gradient: NDArray

        self.biases: NDArray
        self.biases_gradient: NDArray

        if input_size is not None:
            self.build(input_size)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {"output_size": self.output_shape, "input_size": self.input_shape, "initialization": self.initialization, "no_bias": self.no_bias}
        )
        return config

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)

        if self.initialization != "auto":
            self.init_weights(self.initialization, self.no_bias)

    def init_weights(self, method: Lit_W, no_bias: bool) -> None:
        std_dev = 0.1

        if method == "he":
            std_dev = np.sqrt(2.0 / self.input_size)
        elif method == "xavier":
            std_dev = np.sqrt(1.0 / self.input_size)

        self.weights = np.random.randn(self.input_size, self.output_size) * std_dev
        self.weights_gradient = np.zeros_like(self.weights)

        self.no_bias = no_bias

        if not self.no_bias:
            self.biases = np.random.randn(1, self.output_size)
            self.biases_gradient = np.zeros_like(self.biases)

    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        self.input = input_batch

        res: NDArray
        if self.no_bias:
            res = self.input @ self.weights
        else:
            res = self.input @ self.weights + self.biases
        return res

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        self.weights_gradient = self.input.T @ output_gradient_batch
        if not self.no_bias:
            self.biases_gradient = np.sum(output_gradient_batch, axis=0, keepdims=True)

        grad: NDArray = output_gradient_batch @ self.weights.T
        return grad

    @property
    def params(self) -> dict[str, tuple[NDArray, NDArray]]:
        params = {"weights": (self.weights, self.weights_gradient)}
        if not self.no_bias:
            params["biases"] = (self.biases, self.biases_gradient)
        return params

    def load_params(self, params: dict[str, NDArray]) -> None:
        self.weights[:] = params["weights"]
        if not self.no_bias:
            self.biases[:] = params["biases"]


class Dropout(Layer):
    def __init__(self, probability: float = 0.5) -> None:
        super().__init__()
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
        super().__init__()
        self.momentum: float = momentum
        self.epsilon: float = epsilon

        self.gamma: NDArray
        self.beta: NDArray

        self.cache_m: NDArray
        self.cache_v: NDArray

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)

        self.gamma = np.ones((1, self.input_size))
        self.gamma_gradient = np.zeros_like(self.gamma)

        self.beta = np.zeros((1, self.input_size))
        self.beta_gradient = np.zeros_like(self.beta)

        self.cache_m = np.zeros((1, self.input_size))
        self.cache_v = np.ones((1, self.input_size))

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
        self.gamma[:] = params["gamma"]
        self.beta[:] = params["beta"]

    @property
    def state(self) -> dict[str, NDArray]:
        return {"cache_m": self.cache_m, "cache_v": self.cache_v}

    @state.setter
    def state(self, state: dict[str, NDArray]) -> None:
        self.cache_m = state["cache_m"]
        self.cache_v = state["cache_v"]


class Convolutional(Layer):
    def __init__(
        self, output_depth: int, kernel_size: int, input_shape: tuple | None = None, initialization: Lit_W = "auto", no_bias: bool = False
    ) -> None:
        super().__init__()
        self.output_depth: int = output_depth
        self.kernel_size: int = kernel_size
        self.initialization: Lit_W = initialization
        self.no_bias: bool = no_bias

        self.X_col: NDArray
        self.kernels: NDArray
        self.kernels_gradient: NDArray
        self.biases: NDArray
        self.biases_gradient: NDArray

        if input_shape is not None:
            self.build(input_shape)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "output_depth": self.output_depth,
                "kernel_size": self.kernel_size,
                "input_shape": self.input_shape,
                "initialization": self.initialization,
                "no_bias": self.no_bias,
            }
        )
        return config

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)

        _, input_height, input_width = self.input_shape

        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1
        self.output_shape = (self.output_depth, output_height, output_width)
        self.output_shape = self.output_shape

        if self.initialization != "auto":
            self.init_weights(self.initialization, self.no_bias)

    def init_weights(self, method: Lit_W, no_bias: bool) -> None:
        std_dev = 0.1

        input_depth, _, _ = self.input_shape

        if method == "he":
            std_dev = np.sqrt(2.0 / (input_depth * self.kernel_size * self.kernel_size))
        elif method == "xavier":
            std_dev = np.sqrt(1.0 / (input_depth * self.kernel_size * self.kernel_size))

        kernels_shape = (self.output_depth, input_depth, self.kernel_size, self.kernel_size)

        self.kernels = np.random.randn(*kernels_shape) * std_dev
        self.kernels_gradient = np.zeros_like(self.kernels)

        self.no_bias = no_bias

        if not self.no_bias:
            self.biases = np.random.randn(self.output_depth)
            self.biases_gradient = np.zeros_like(self.biases)

    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        self.input = input_batch
        batch_size = input_batch.shape[0]

        self.X_col = self._im2col(input_batch)
        k_col = self.kernels.reshape(self.output_depth, -1)

        _, output_height, output_width = self.output_shape

        output = self.X_col @ k_col.T
        if not self.no_bias:
            output += self.biases
        output = output.reshape(batch_size, output_height, output_width, self.output_depth)
        return output.transpose(0, 3, 1, 2)  # type: ignore[no-any-return]

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        dOut_col = output_gradient_batch.transpose(0, 2, 3, 1).reshape(-1, self.output_depth)

        if not self.no_bias:
            self.biases_gradient = np.sum(dOut_col, axis=0)

        k_col = dOut_col.T @ self.X_col
        self.kernels_gradient = k_col.reshape(self.output_depth, self.input_shape[0], self.kernel_size, self.kernel_size)

        W_col = self.kernels.reshape(self.output_depth, -1)
        dX_col = dOut_col @ W_col

        return self._col2im(dX_col, self.input.shape)

    def _col2im(self, cols: NDArray, input_shape: tuple) -> NDArray:
        N, C, _, _ = input_shape
        _, H_out, W_out = self.output_shape
        K = self.kernel_size

        cols_reshaped = cols.reshape(N, H_out, W_out, C, K, K)

        cols_transposed = cols_reshaped.transpose(0, 3, 1, 2, 4, 5)

        dX = np.zeros(input_shape)

        for i in range(K):
            for j in range(K):
                dX[:, :, i : i + H_out, j : j + W_out] += cols_transposed[:, :, :, :, i, j]

        return dX

    def _im2col(self, input_batch: NDArray) -> NDArray:
        _, C, _, _ = input_batch.shape
        K = self.kernel_size

        window = sliding_window_view(input_batch, window_shape=(K, K), axis=(2, 3))  # type: ignore[call-overload]
        window = window.transpose(0, 2, 3, 1, 4, 5)
        return window.reshape(-1, C * K * K)  # type: ignore[no-any-return]

    @property
    def params(self) -> dict[str, tuple[NDArray, NDArray]]:
        return {"kernels": (self.kernels, self.kernels_gradient), "biases": (self.biases, self.biases_gradient)}

    def load_params(self, params: dict[str, NDArray]) -> None:
        self.kernels[:] = params["kernels"]
        self.biases[:] = params["biases"]


class Flatten(Layer):
    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        return input_batch.reshape(input_batch.shape[0], -1)

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        return output_gradient_batch.reshape(output_gradient_batch.shape[0], *self.input_shape)
