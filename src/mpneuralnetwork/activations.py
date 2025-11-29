from collections.abc import Callable

from . import DTYPE, ArrayType, xp
from .layers import Layer

T = Callable[[ArrayType], ArrayType]


class Activation(Layer):
    def __init__(self, activation: T, activation_prime: T) -> None:
        self.activation: T = activation
        self.activation_prime: T = activation_prime

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        self.input = input_batch
        return self.activation(self.input)

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        res: ArrayType = xp.multiply(output_gradient_batch, self.activation_prime(self.input))
        return res

    @property
    def params(self) -> dict[str, tuple[ArrayType, ArrayType]]:
        return {}


class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(
            lambda x: xp.tanh(x, dtype=DTYPE),
            lambda x: (1 - xp.tanh(x, dtype=DTYPE) ** 2),
        )


class Sigmoid(Activation):
    def __init__(self) -> None:
        def sigmoid(x: ArrayType) -> ArrayType:
            return 1 / (1 + xp.exp(-x, dtype=DTYPE))  # type: ignore[no-any-return]

        super().__init__(lambda x: sigmoid(x), lambda x: sigmoid(x) * (1 - sigmoid(x)))


class ReLU(Activation):
    def __init__(self) -> None:
        super().__init__(lambda x: xp.maximum(0, x, dtype=DTYPE), lambda x: x > 0)


class PReLU(Activation):
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__(
            lambda x: xp.maximum(alpha * x, x, dtype=DTYPE),
            lambda x: xp.where(x < 0, alpha, 1),
        )
        self.alpha: float = alpha

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


class Swish(Activation):
    def __init__(self) -> None:
        super().__init__(
            lambda x: x / (1 + xp.exp(-x, dtype=DTYPE)),
            lambda x: (1 + xp.exp(-x, dtype=DTYPE) + x * xp.exp(-x, dtype=DTYPE)) / (1 + xp.exp(-x, dtype=DTYPE)) ** 2,
        )


class Softmax(Layer):
    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        m = xp.max(input_batch, axis=1, keepdims=True)
        e = xp.exp(input_batch - m, dtype=DTYPE)
        self.output = e / xp.sum(e, axis=1, keepdims=True, dtype=DTYPE)
        return self.output

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        sum_s_times_g: ArrayType = xp.sum(self.output * output_gradient_batch, axis=1, keepdims=True, dtype=DTYPE)  # type: ignore[assignment]

        res: ArrayType = self.output * (output_gradient_batch - sum_s_times_g)
        return res

    @property
    def params(self) -> dict[str, tuple[ArrayType, ArrayType]]:
        return {}
