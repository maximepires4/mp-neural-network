from abc import abstractmethod
from typing import Literal

from . import DTYPE, ArrayType, xp
from .layers import Layer

T = dict[int, ArrayType]
Lit_R = Literal["L1", "L2"]


class Optimizer:
    """Base class for all optimization algorithms.

    Optimizers update the weights of the network layers to minimize the loss function.
    They also handle regularization (L1/L2).
    """

    def __init__(self, learning_rate: float, regularization: Lit_R, weight_decay: float) -> None:
        """Initializes the optimizer.

        Args:
            learning_rate (float): The step size for parameter updates.
            regularization (Lit_R): Type of regularization ('L1' or 'L2').
            weight_decay (float): The strength of the regularization (lambda).
        """
        self.learning_rate: float = learning_rate
        self.regularization: Lit_R = regularization
        self.weight_decay: float = weight_decay

    @abstractmethod
    def step(self, layers: list[Layer]) -> None:
        """Performs a single optimization step.

        Iterates over all layers and updates their parameters based on stored gradients.

        Args:
            layers (list[Layer]): List of layers containing parameters to update.
        """
        pass

    def get_config(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "weight_decay": self.weight_decay,
        }

    def apply_regularization(self, param_name: str, param: ArrayType) -> ArrayType | int:
        """Computes the regularization gradient term.

        Args:
            param_name (str): Name of the parameter (e.g., 'weights', 'bias').
            param (ArrayType): The parameter value.

        Returns:
            ArrayType | int: The gradient contribution from regularization.
        """
        regularization: ArrayType
        if "bias" in param_name or "beta" in param_name or "gamma" in param_name:
            return 0

        if self.regularization == "L2":
            regularization = self.weight_decay * param
        else:
            regularization = self.weight_decay * xp.sign(param)

        return regularization

    @property
    def params(self) -> dict:
        """Returns the optimizer's internal state (velocities, moments)."""
        return {}


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) with Momentum.

    Update rule:
        1. `v = momentum * v - lr * gradient`
        2. `w = w + v`

    Args:
        learning_rate (float, optional): Step size. Defaults to 0.01.
        regularization (Lit_R, optional): 'L1' or 'L2'. Defaults to 'L2'.
        weight_decay (float, optional): Regularization strength. Defaults to 0.001.
        momentum (float, optional): Momentum factor (0 to 1). Defaults to 0.1.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        regularization: Lit_R = "L2",
        weight_decay: float = 0.001,
        momentum: float = 0.1,
    ) -> None:
        super().__init__(learning_rate, regularization, weight_decay)
        self.momentum: float = momentum

        self.velocities: T = {}

    def step(self, layers: list[Layer]) -> None:
        for layer in layers:
            if not hasattr(layer, "params"):
                continue

            for param_name, (param, grad) in layer.params.items():
                grad += self.apply_regularization(param_name, param)

                p_id: int = id(param)

                if p_id not in self.velocities:
                    self.velocities[p_id] = xp.zeros_like(param, dtype=DTYPE)

                # Velocity Update: v = momentum * v - lr * grad

                # 1. v *= momentum (in-place)
                xp.multiply(self.velocities[p_id], self.momentum, out=self.velocities[p_id])

                # 2. v -= lr * grad
                self.velocities[p_id] -= self.learning_rate * grad

                # Parameter Update: w += v
                param += self.velocities[p_id]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"momentum": self.momentum})
        return config

    @property
    def params(self) -> dict:
        return {"velocities": self.velocities}


class RMSprop(Optimizer):
    """RMSprop optimizer.

    Adapts learning rates by dividing the gradient by a running average of its recent magnitude.

    Update rule:
        1. `cache = decay * cache + (1 - decay) * grad^2`
        2. `w = w - lr * grad / (sqrt(cache) + epsilon)`

    Args:
        learning_rate (float): Defaults to 0.001.
        regularization (Lit_R): 'L1' or 'L2'.
        weight_decay (float): Defaults to 0.001.
        decay_rate (float, optional): Discounting factor. Defaults to 0.9.
        epsilon (float, optional): Small value for numerical stability. Defaults to 1e-8.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        regularization: Lit_R = "L2",
        weight_decay: float = 0.001,
        decay_rate: float = 0.9,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate, regularization, weight_decay)
        self.decay_rate: float = decay_rate
        self.epsilon: float = epsilon

        self.cache: T = {}

    def step(self, layers: list[Layer]) -> None:
        for layer in layers:
            if not hasattr(layer, "params"):
                continue

            for param_name, (param, grad) in layer.params.items():
                grad += self.apply_regularization(param_name, param)

                p_id: int = id(param)

                if p_id not in self.cache:
                    self.cache[p_id] = xp.zeros_like(param, dtype=DTYPE)

                # Cache Update: cache = decay * cache + (1 - decay) * grad^2

                # 1. cache *= decay (in-place)
                xp.multiply(self.cache[p_id], self.decay_rate, out=self.cache[p_id])

                # 2. cache += (1 - decay) * grad^2
                self.cache[p_id] += (1 - self.decay_rate) * xp.square(grad)

                # Parameter Update: w -= lr * grad / (sqrt(cache) + epsilon)

                # 1. Denominator = sqrt(cache) + epsilon
                denom = xp.sqrt(self.cache[p_id])
                xp.add(denom, self.epsilon, out=denom)

                # 2. Update = lr * grad / denom
                # w -= update
                param -= self.learning_rate * grad / denom

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"decay_rate": self.decay_rate, "epsilon": self.epsilon})
        return config

    @property
    def params(self) -> dict:
        return {"cache": self.cache}


class Adam(Optimizer):
    """Adam Optimizer (Adaptive Moment Estimation).

    Combines Momentum and RMSprop.
    Implements **Decoupled Weight Decay (AdamW)** when `regularization='L2'`.

    Update rule:
        1. `m = beta1 * m + (1 - beta1) * g`
        2. `v = beta2 * v + (1 - beta2) * g^2`
        3. `m_hat = m / (1 - beta1^t)`
        4. `v_hat = v / (1 - beta2^t)`
        5. `w = w - lr * m_hat / (sqrt(v_hat) + eps)`
        6. If L2: `w = w - lr * decay * w` (Decoupled)

    Args:
        beta1 (float, optional): Decay rate for first moment. Defaults to 0.9.
        beta2 (float, optional): Decay rate for second moment. Defaults to 0.999.
        epsilon (float, optional): Stability term. Defaults to 1e-8.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        regularization: Lit_R = "L2",
        weight_decay: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate, regularization, weight_decay)
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon

        self.t: int = 0
        self.momentums: T = {}
        self.velocities: T = {}

    def step(self, layers: list[Layer]) -> None:
        self.t += 1

        for layer in layers:
            if not hasattr(layer, "params"):
                continue

            for param_name, (param, grad) in layer.params.items():
                if self.regularization == "L1":
                    grad += self.apply_regularization(param_name, param)

                p_id: int = id(param)

                if p_id not in self.momentums:
                    self.momentums[p_id] = xp.zeros_like(param, dtype=DTYPE)
                    self.velocities[p_id] = xp.zeros_like(param, dtype=DTYPE)

                # --- 1. Update Momentum (First Moment) ---
                # m = beta1 * m + (1 - beta1) * grad

                # m *= beta1
                xp.multiply(self.momentums[p_id], self.beta1, out=self.momentums[p_id])
                # m += (1 - beta1) * grad
                self.momentums[p_id] += (1 - self.beta1) * grad

                # --- 2. Update Velocity (Second Moment) ---
                # v = beta2 * v + (1 - beta2) * grad^2

                # v *= beta2
                xp.multiply(self.velocities[p_id], self.beta2, out=self.velocities[p_id])
                # v += (1 - beta2) * grad^2
                self.velocities[p_id] += (1 - self.beta2) * xp.square(grad)

                # --- 3. Bias Correction ---
                # m_hat = m / (1 - beta1^t)
                # v_hat = v / (1 - beta2^t)

                bias_correction1 = 1 - self.beta1**self.t
                bias_correction2 = 1 - self.beta2**self.t

                # Efficient Update Formula:
                # w -= lr * m_hat / (sqrt(v_hat) + epsilon)
                # w -= (lr / bias_correction1) * m / (sqrt(v / bias_correction2) + epsilon)

                step_size = self.learning_rate / bias_correction1

                # Denominator construction
                denom = xp.sqrt(self.velocities[p_id])
                # denom /= sqrt(bias_correction2)
                xp.divide(denom, xp.sqrt(bias_correction2), out=denom)
                # denom += epsilon
                xp.add(denom, self.epsilon, out=denom)

                # Final update: w -= step_size * m / denom
                # param -= step_size * (self.momentums[p_id] / denom)
                param -= step_size * self.momentums[p_id] / denom

                if self.regularization == "L2":
                    param -= self.learning_rate * self.apply_regularization(param_name, param)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
            }
        )
        return config

    @property
    def params(self) -> dict:
        return {
            "t": self.t,
            "momentums": self.momentums,
            "velocities": self.velocities,
        }
