from .. import DTYPE, ArrayType, xp
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

        self.weights: ArrayType
        self.weights_gradient: ArrayType

        self.biases: ArrayType
        self.biases_gradient: ArrayType

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
            std_dev = xp.sqrt(2.0 / self.input_size, dtype=DTYPE)
        elif method == "xavier":
            std_dev = xp.sqrt(1.0 / self.input_size, dtype=DTYPE)

        self.weights = xp.random.randn(self.input_size, self.output_size).astype(DTYPE) * std_dev
        self.weights_gradient = xp.zeros_like(self.weights, dtype=DTYPE)

        self.no_bias = no_bias

        if not self.no_bias:
            self.biases = xp.random.randn(1, self.output_size).astype(DTYPE)
            self.biases_gradient = xp.zeros_like(self.biases, dtype=DTYPE)

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        self.input = input_batch

        res: ArrayType = self.input @ self.weights
        if not self.no_bias:
            res += self.biases
        return res

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        self.weights_gradient = self.input.T @ output_gradient_batch
        if not self.no_bias:
            self.biases_gradient = xp.sum(output_gradient_batch, axis=0, keepdims=True, dtype=DTYPE)  # type: ignore

        grad: ArrayType = output_gradient_batch @ self.weights.T
        return grad

    @property
    def params(self) -> dict[str, tuple[ArrayType, ArrayType]]:
        params = {"weights": (self.weights, self.weights_gradient)}
        if not self.no_bias:
            params["biases"] = (self.biases, self.biases_gradient)
        return params

    def load_params(self, params: dict[str, ArrayType]) -> None:
        self.weights[:] = params["weights"]
        if not self.no_bias:
            self.biases[:] = params["biases"]


class Dropout(Layer):
    def __init__(self, probability: float = 0.5) -> None:
        super().__init__()
        self.probability: float = probability
        self.mask: ArrayType

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"probability": self.probability})
        return config

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        if not training:
            return input_batch

        self.mask = xp.random.binomial(1, 1 - self.probability, size=input_batch.shape).astype(DTYPE) / (1 - self.probability)

        res: ArrayType = input_batch * self.mask
        return res

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        grad: ArrayType = output_gradient_batch * self.mask
        return grad


class BatchNormalization(Layer):
    def __init__(self, momentum: float = 0.9, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.momentum: float = momentum
        self.epsilon: float = epsilon

        self.gamma: ArrayType
        self.beta: ArrayType

        self.cache_m: ArrayType
        self.cache_v: ArrayType

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)

        self.gamma = xp.ones((1, self.input_size), dtype=DTYPE)
        self.gamma_gradient = xp.zeros_like(self.gamma, dtype=DTYPE)

        self.beta = xp.zeros((1, self.input_size), dtype=DTYPE)
        self.beta_gradient = xp.zeros_like(self.beta, dtype=DTYPE)

        self.cache_m = xp.zeros((1, self.input_size), dtype=DTYPE)
        self.cache_v = xp.ones((1, self.input_size), dtype=DTYPE)

        self.std_inv: ArrayType
        self.x_centered: ArrayType
        self.x_norm: ArrayType

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"momentum": self.momentum, "epsilon": self.epsilon})
        return config

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        self.input = input_batch

        mean: ArrayType
        var: ArrayType

        if training:
            mean = xp.mean(self.input, axis=0, keepdims=True, dtype=DTYPE)  # type: ignore
            var = xp.var(self.input, axis=0, keepdims=True, dtype=DTYPE)

            self.cache_m = self.momentum * self.cache_m + (1 - self.momentum) * mean
            self.cache_v = self.momentum * self.cache_v + (1 - self.momentum) * var

        else:
            mean = self.cache_m
            var = self.cache_v

        self.std_inv = 1 / xp.sqrt(var + self.epsilon, dtype=DTYPE)
        self.x_centered = self.input - mean
        self.x_norm = self.x_centered * self.std_inv

        res: ArrayType = self.x_norm * self.gamma
        res += self.beta
        return res

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        self.gamma_gradient = xp.sum(self.x_norm * output_gradient_batch, axis=0, keepdims=True, dtype=DTYPE)  # type: ignore
        self.beta_gradient = xp.sum(output_gradient_batch, axis=0, keepdims=True, dtype=DTYPE)  # type: ignore

        N = output_gradient_batch.shape[0]
        dx_norm = output_gradient_batch * self.gamma

        grad: ArrayType = (
            (1 / N)
            * self.std_inv
            * (
                N * dx_norm
                - xp.sum(dx_norm, axis=0, keepdims=True, dtype=DTYPE)
                - self.x_norm * xp.sum(dx_norm * self.x_norm, axis=0, keepdims=True, dtype=DTYPE)
            )
        )
        return grad

    @property
    def params(self) -> dict[str, tuple[ArrayType, ArrayType]]:
        return {  # type: ignore
            "gamma": (self.gamma, self.gamma_gradient),
            "beta": (self.beta, self.beta_gradient),
        }

    def load_params(self, params: dict[str, ArrayType]) -> None:
        self.gamma[:] = params["gamma"]
        self.beta[:] = params["beta"]

    @property
    def state(self) -> dict[str, ArrayType]:
        return {"cache_m": self.cache_m, "cache_v": self.cache_v}

    @state.setter
    def state(self, state: dict[str, ArrayType]) -> None:
        self.cache_m = state["cache_m"]
        self.cache_v = state["cache_v"]
