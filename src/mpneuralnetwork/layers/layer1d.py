from .. import DTYPE, ArrayType, xp
from .layer import Layer, Lit_W


class Dense(Layer):
    """Fully Connected (Dense) Layer.

    Every neuron in the input is connected to every neuron in the output.

    Operation:
        `Y = X @ W + b`

    Attributes:
        output_size (int): Dimensionality of the output space.
        input_size (int): Dimensionality of the input space.
        initialization (Lit_W): Weight initialization method ("auto", "he", "xavier").
        no_bias (bool): Whether to disable the bias vector.
        weights (ArrayType): Weight matrix of shape (input_size, output_size).
        biases (ArrayType): Bias vector of shape (1, output_size).
    """

    def __init__(
        self,
        output_size: int,
        input_size: int | None = None,
        initialization: Lit_W = "auto",
        no_bias: bool = False,
    ) -> None:
        """Initializes the Dense layer.

        Args:
            output_size (int): Number of neurons in this layer.
            input_size (int | None, optional): Number of input features. If None, inferred at build time.
            initialization (Lit_W, optional): Weight init strategy. Defaults to "auto".
            no_bias (bool, optional): If True, bias is not used. Defaults to False.
        """
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
        """Initializes weights using the specified method.

        Args:
            method (Lit_W): Initialization method.
                - "he": Kaiming He initialization (for ReLU).
                - "xavier": Xavier Glorot initialization (for Sigmoid/Tanh).
            no_bias (bool): Whether to disable bias (e.g. if followed by BatchNorm).
        """
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
        """Performs forward propagation.

        Args:
            input_batch (ArrayType): Input data of shape (batch_size, input_size).
            training (bool, optional): Unused for Dense layer. Defaults to True.

        Returns:
            ArrayType: Output data of shape (batch_size, output_size).
        """
        self.input = input_batch

        res: ArrayType = self.input @ self.weights
        if not self.no_bias:
            res += self.biases
        return res

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        """Performs backward propagation.

        Computes gradients for weights, biases, and inputs.

        Args:
            output_gradient_batch (ArrayType): Gradient w.r.t output (batch_size, output_size).

        Returns:
            ArrayType: Gradient w.r.t input (batch_size, input_size).
        """
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
    """Dropout Layer for regularization.

    Randomly sets input units to 0 with a frequency of `probability` at each step during training time,
    which helps prevent overfitting.

    Training:
        `output = input * mask` (where mask is Bernoulli(1-p))
        Values are scaled by `1/(1-p)` to preserve magnitude.

    Inference:
        `output = input` (Identity function).

    Attributes:
        probability (float): The dropout rate (fraction of input units to drop).
    """

    def __init__(self, probability: float = 0.5) -> None:
        """Initializes Dropout.

        Args:
            probability (float, optional): Fraction of the input units to drop. Defaults to 0.5.
        """
        super().__init__()
        self.probability: float = probability
        self.mask: ArrayType

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"probability": self.probability})
        return config

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        """Applies dropout to the input.

        Args:
            input_batch (ArrayType): Input data.
            training (bool, optional): If True, applies random dropout. If False, returns input as is.

        Returns:
            ArrayType: Processed input.
        """
        if not training:
            return input_batch

        self.mask = xp.random.binomial(1, 1 - self.probability, size=input_batch.shape).astype(DTYPE) / (1 - self.probability)

        res: ArrayType = input_batch * self.mask
        return res

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        """Propagates gradients through the dropout mask.

        Args:
            output_gradient_batch (ArrayType): Gradient from next layer.

        Returns:
            ArrayType: Gradient w.r.t input (zeroed out where inputs were dropped).
        """
        grad: ArrayType = output_gradient_batch * self.mask
        return grad


class BatchNormalization(Layer):
    """Batch Normalization Layer (1D).

    Normalize the activations of the previous layer at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation standard deviation close to 1.

    Training:
        Uses batch statistics (mean, variance) to normalize. Updates running moving averages.

    Inference:
        Uses learned running statistics (cache_m, cache_v) to normalize.

    Attributes:
        momentum (float): Momentum for the moving average updating.
        epsilon (float): Small float added to variance to avoid dividing by zero.
        gamma (ArrayType): Learnable scale parameter.
        beta (ArrayType): Learnable shift parameter.
    """

    def __init__(self, momentum: float = 0.9, epsilon: float = 1e-8) -> None:
        """Initializes BatchNormalization.

        Args:
            momentum (float, optional): Momentum for moving average (typically 0.9 or 0.99). Defaults to 0.9.
            epsilon (float, optional): Epsilon for stability. Defaults to 1e-8.
        """
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
        """Performs batch normalization.

        Args:
            input_batch (ArrayType): Input data of shape (batch_size, input_size).
            training (bool, optional): If True, uses batch stats and updates running averages.
                If False, uses running averages.

        Returns:
            ArrayType: Normalized and scaled data.
        """
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
        """Computes gradients for BN.

        Args:
            output_gradient_batch (ArrayType): Gradient w.r.t output.

        Returns:
            ArrayType: Gradient w.r.t input.
        """
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
