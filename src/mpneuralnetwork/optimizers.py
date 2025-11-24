from abc import abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .layers import Layer

T = dict[int, NDArray]
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

    def apply_regularization(self, param_name, param) -> NDArray | int:
        regularization: NDArray
        if "bias" in param_name or "beta" in param_name or "gamma" in param_name:
            return 0

        if self.regularization == "L2":
            regularization = self.weight_decay * param
        else:
            regularization = self.weight_decay * np.sign(param)

        return regularization

    @property
    def params(self) -> dict:
        return {}


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, regularization: Lit_R = "L2", weight_decay: float = 0.001, momentum: float = 0.1) -> None:
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
                    self.velocities[p_id] = np.zeros_like(param)

                self.velocities[p_id] = self.momentum * self.velocities[p_id] - self.learning_rate * grad
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
        self, learning_rate: float = 0.001, regularization: Lit_R = "L2", weight_decay: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8
    ) -> None:
        super().__init__(learning_rate, regularization, weight_decay)
        self.decay_rate: float = decay_rate
        self.epsilon: float = epsilon

        self.cache: T = {}

    def step(self, layers) -> None:
        for layer in layers:
            if not hasattr(layer, "params"):
                continue

            for param_name, (param, grad) in layer.params.items():
                grad += self.apply_regularization(param_name, param)

                p_id: int = id(param)

                if p_id not in self.cache:
                    self.cache[p_id] = np.zeros_like(param)

                self.cache[p_id] = self.decay_rate * self.cache[p_id] + (1 - self.decay_rate) * np.power(grad, 2)

                param -= self.learning_rate * grad / np.sqrt(self.cache[p_id] + self.epsilon)

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

    def step(self, layers) -> None:
        self.t += 1

        for layer in layers:
            if not hasattr(layer, "params"):
                continue

            for param_name, (param, grad) in layer.params.items():
                if self.regularization == "L1":
                    grad += self.apply_regularization(param_name, param)

                p_id: int = id(param)

                if p_id not in self.momentums:
                    self.momentums[p_id] = np.zeros_like(param)
                    self.velocities[p_id] = np.zeros_like(param)

                self.momentums[p_id] = self.beta1 * self.momentums[p_id] + (1 - self.beta1) * grad
                self.velocities[p_id] = self.beta2 * self.velocities[p_id] + (1 - self.beta2) * np.power(grad, 2)

                m_hat = self.momentums[p_id] / (1 - self.beta1**self.t)
                v_hat = self.velocities[p_id] / (1 - self.beta2**self.t)

                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

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
