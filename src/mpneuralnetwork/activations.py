from collections.abc import Callable

from . import DTYPE, ArrayType, xp
from .layers import Layer

T = Callable[[ArrayType], ArrayType]


class Activation(Layer):
    """Base class for activation functions.

    Activations are treated as layers in this framework. They apply a non-linear
    transformation element-wise to the input.

    Attributes:
        activation (Callable): The function to apply during the forward pass.
        activation_prime (Callable): The derivative of the function for the backward pass.
    """

    def __init__(self, activation: T, activation_prime: T) -> None:
        """Initializes the activation layer.

        Args:
            activation (Callable[[ArrayType], ArrayType]): The activation function.
            activation_prime (Callable[[ArrayType], ArrayType]): The derivative of the activation function.
        """
        self.activation: T = activation
        self.activation_prime: T = activation_prime

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        """Applies the activation function to the input.

        Args:
            input_batch (ArrayType): Input data of any shape.
            training (bool, optional): Whether in training mode. Defaults to True.

        Returns:
            ArrayType: Activated output with the same shape as `input_batch`.
        """
        self.input = input_batch
        return self.activation(self.input)

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        """Computes the gradient of the activation function.

        Applies the chain rule: `grad = output_gradient * activation'(input)`.

        Args:
            output_gradient_batch (ArrayType): Gradient flowing from the next layer.

        Returns:
            ArrayType: Gradient with respect to the input.
        """
        res: ArrayType = xp.multiply(output_gradient_batch, self.activation_prime(self.input))
        return res

    @property
    def params(self) -> dict[str, tuple[ArrayType, ArrayType]]:
        """Activations usually have no trainable parameters.

        Returns:
            dict: Empty dictionary.
        """
        return {}


class Tanh(Activation):
    """Hyperbolic Tangent activation function.

    Formula:
        `f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

    Range: (-1, 1).
    Zero-centered, making it often preferable to Sigmoid for hidden layers.
    """

    def __init__(self) -> None:
        super().__init__(
            lambda x: xp.tanh(x, dtype=DTYPE),
            lambda x: (1 - xp.tanh(x, dtype=DTYPE) ** 2),
        )


class Sigmoid(Activation):
    """Sigmoid activation function.

    Formula:
        `f(x) = 1 / (1 + exp(-x))`

    Range: (0, 1).
    Used for binary classification (output layer) or gating mechanisms (like in LSTMs).
    Can suffer from vanishing gradients in deep networks.
    """

    def __init__(self) -> None:
        def sigmoid(x: ArrayType) -> ArrayType:
            return 1 / (1 + xp.exp(-x, dtype=DTYPE))  # type: ignore[no-any-return]

        super().__init__(lambda x: sigmoid(x), lambda x: sigmoid(x) * (1 - sigmoid(x)))


class ReLU(Activation):
    """Rectified Linear Unit activation function.

    Formula:
        `f(x) = max(0, x)`

    Range: [0, inf).
    Computationally efficient and mitigates the vanishing gradient problem.
    Most common activation for hidden layers in deep networks.
    """

    def __init__(self) -> None:
        super().__init__(lambda x: xp.maximum(0, x, dtype=DTYPE), lambda x: x > 0)


class PReLU(Activation):
    """Parametric Rectified Linear Unit.

    Formula:
        `f(x) = x` if `x > 0`
        `f(x) = alpha * x` if `x <= 0`

    Where `alpha` is a learnable parameter updated during training.
    Allows the network to learn the negative slope, avoiding "dying ReLU" problems.

    Args:
        alpha (float, optional): Initial value for the negative slope. Defaults to 0.01.
    """

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
    """Swish activation function.

    Formula:
        `f(x) = x * sigmoid(x)`

    Range: (~-0.28, inf).
    Proposed by Google. A smooth, non-monotonic function that often outperforms ReLU
    on deep networks.
    """

    def __init__(self) -> None:
        super().__init__(
            lambda x: x / (1 + xp.exp(-x, dtype=DTYPE)),
            lambda x: (1 + xp.exp(-x, dtype=DTYPE) + x * xp.exp(-x, dtype=DTYPE)) / (1 + xp.exp(-x, dtype=DTYPE)) ** 2,
        )


class Softmax(Layer):
    """Softmax activation function.

    Formula:
        `f(x)_i = exp(x_i / T) / sum(exp(x_j / T))`

    Typically used in the output layer for multi-class classification.
    Converts a vector of K real numbers into a probability distribution of K possible outcomes.
    The temperature parameter T is used to scale the logits before computing the softmax.
    """

    def __init__(self, temperature: float = 1.0, epsilon: float = 1e-8) -> None:
        """Initializes the Softmax layer.

        Args:
            temperature (float, optional): Temperature parameter. Defaults to 1.0.
            epsilon (float, optional): Small float added to denominator to avoid dividing by zero. Defaults to 1e-8.
        """
        self.temperature: float = temperature
        self.epsilon: float = epsilon

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        """Applies Softmax function.

        Args:
            input_batch (ArrayType): Input logits of shape (batch_size, num_classes).
            training (bool, optional): Unused. Defaults to True.

        Returns:
            ArrayType: Probabilities of shape (batch_size, num_classes).
        """
        scaled_logits = input_batch / (self.temperature + self.epsilon)

        m = xp.max(scaled_logits, axis=1, keepdims=True)
        e = xp.exp(scaled_logits - m, dtype=DTYPE)

        self.output = e / xp.sum(e, axis=1, keepdims=True, dtype=DTYPE)
        return self.output

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        """Computes gradient for Softmax.

        Note: This is rarely used directly if using `CategoricalCrossEntropy` loss,
        as the framework optimizes the combined gradient calculation for numerical stability.

        Args:
            output_gradient_batch (ArrayType): Gradient from next layer.

        Returns:
            ArrayType: Gradient w.r.t input.
        """
        sum_s_times_g: ArrayType = xp.sum(self.output * output_gradient_batch, axis=1, keepdims=True, dtype=DTYPE)  # type: ignore[assignment]

        res: ArrayType = (self.output * (output_gradient_batch - sum_s_times_g)) / (self.temperature + self.epsilon)
        return res

    @property
    def params(self) -> dict[str, tuple[ArrayType, ArrayType]]:
        return {}
