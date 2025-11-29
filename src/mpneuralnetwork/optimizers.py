from abc import abstractmethod
from typing import Literal

from . import DTYPE, ArrayType, xp
from .layers import Layer

T = dict[int, ArrayType]
Lit_R = Literal["L1", "L2"]


class Optimizer:
    def __init__(self, learning_rate: float, regularization: Lit_R, weight_decay: float) -> None:
        self.learning_rate: float = learning_rate
        self.regularization: Lit_R = regularization
        self.weight_decay: float = weight_decay

    @abstractmethod
    def step(self, layers: list[Layer]) -> None:
        pass

    def get_config(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "weight_decay": self.weight_decay,
        }

    def apply_regularization(self, param_name: str, param: ArrayType) -> ArrayType | int:
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
        return {}


class SGD(Optimizer):
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
