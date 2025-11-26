import numpy as np
from numpy.typing import NDArray

from .. import DTYPE
from .layer import Layer, Lit_W


class Dense(Layer):
    def __init__(
        self,
        output_size: int,
        input_size: int | None = None,
        initialization: Lit_W = "auto",
        no_bias: bool = False,
    ) -> None:
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
            {
                "output_size": self.output_size,
                "input_size": self.input_size,
                "initialization": self.initialization,
                "no_bias": self.no_bias,
            }
        )
        return config

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)

        if self.initialization != "auto":
            self.init_weights(self.initialization, self.no_bias)

    def init_weights(self, method: Lit_W, no_bias: bool) -> None:
        std_dev = 0.1

        if method == "he":
            std_dev = np.sqrt(2.0 / self.input_size, dtype=DTYPE)
        elif method == "xavier":
            std_dev = np.sqrt(1.0 / self.input_size, dtype=DTYPE)

        self.weights = np.random.randn(self.input_size, self.output_size).astype(DTYPE) * std_dev
        self.weights_gradient = np.zeros_like(self.weights, dtype=DTYPE)

        self.no_bias = no_bias

        if not self.no_bias:
            self.biases = np.random.randn(1, self.output_size).astype(DTYPE)
            self.biases_gradient = np.zeros_like(self.biases, dtype=DTYPE)

    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        self.input = input_batch

        res: NDArray = self.input @ self.weights
        if not self.no_bias:
            res += self.biases
        return res

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        self.weights_gradient = self.input.T @ output_gradient_batch
        if not self.no_bias:
            self.biases_gradient = np.sum(output_gradient_batch, axis=0, keepdims=True, dtype=DTYPE)  # type: ignore

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

        self.mask = np.random.binomial(1, 1 - self.probability, size=input_batch.shape).astype(DTYPE) / (1 - self.probability)

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

        self.gamma = np.ones((1, self.input_size), dtype=DTYPE)
        self.gamma_gradient = np.zeros_like(self.gamma, dtype=DTYPE)

        self.beta = np.zeros((1, self.input_size), dtype=DTYPE)
        self.beta_gradient = np.zeros_like(self.beta, dtype=DTYPE)

        self.cache_m = np.zeros((1, self.input_size), dtype=DTYPE)
        self.cache_v = np.ones((1, self.input_size), dtype=DTYPE)

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
            mean = np.mean(self.input, axis=0, keepdims=True, dtype=DTYPE)  # type: ignore
            var = np.var(self.input, axis=0, keepdims=True, dtype=DTYPE)

            self.cache_m = self.momentum * self.cache_m + (1 - self.momentum) * mean
            self.cache_v = self.momentum * self.cache_v + (1 - self.momentum) * var

        else:
            mean = self.cache_m
            var = self.cache_v

        self.std_inv = 1 / np.sqrt(var + self.epsilon, dtype=DTYPE)
        self.x_centered = self.input - mean
        self.x_norm = self.x_centered * self.std_inv

        res: NDArray = self.x_norm * self.gamma
        res += self.beta
        return res

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        self.gamma_gradient = np.sum(self.x_norm * output_gradient_batch, axis=0, keepdims=True, dtype=DTYPE)  # type: ignore
        self.beta_gradient = np.sum(output_gradient_batch, axis=0, keepdims=True, dtype=DTYPE)  # type: ignore

        N = output_gradient_batch.shape[0]
        dx_norm = output_gradient_batch * self.gamma

        grad: NDArray = (
            (1 / N)
            * self.std_inv
            * (
                N * dx_norm
                - np.sum(dx_norm, axis=0, keepdims=True, dtype=DTYPE)
                - self.x_norm * np.sum(dx_norm * self.x_norm, axis=0, keepdims=True, dtype=DTYPE)
            )
        )
        return grad

    @property
    def params(self) -> dict[str, tuple[NDArray, NDArray]]:
        return {  # type: ignore
            "gamma": (self.gamma, self.gamma_gradient),
            "beta": (self.beta, self.beta_gradient),
        }

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
