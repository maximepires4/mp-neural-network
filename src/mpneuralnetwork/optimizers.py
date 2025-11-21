import numpy as np
from numpy.typing import NDArray

from .layers import Layer

T = dict[int, NDArray]


class Optimizer:
    def step(self, layers: list[Layer]) -> None:
        pass

    def get_config(self) -> dict:
        return {"type": self.__class__.__name__}

    @property
    def params(self) -> dict:
        return {}


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.1) -> None:
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum

        self.velocities: T = {}

    def step(self, layers: list[Layer]) -> None:
        for layer in layers:
            if not hasattr(layer, "params"):
                continue

            for _, (param, grad) in layer.params.items():
                p_id: int = id(param)

                if p_id not in self.velocities:
                    self.velocities[p_id] = np.zeros_like(param)

                self.velocities[p_id] = self.momentum * self.velocities[p_id] - self.learning_rate * grad
                param += self.velocities[p_id]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"learning_rate": self.learning_rate, "momentum": self.momentum})
        return config

    @property
    def params(self) -> dict:
        return {"velocities": self.velocities}


class RMSprop(Optimizer):
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8) -> None:
        self.learning_rate: float = learning_rate
        self.decay_rate: float = decay_rate
        self.epsilon: float = epsilon

        self.cache: T = {}

    def step(self, layers) -> None:
        for layer in layers:
            if not hasattr(layer, "params"):
                continue

            for _, (param, grad) in layer.params.items():
                p_id: int = id(param)

                if p_id not in self.cache:
                    self.cache[p_id] = np.zeros_like(param)

                self.cache[p_id] = self.decay_rate * self.cache[p_id] + (1 - self.decay_rate) * np.power(grad, 2)

                param -= self.learning_rate * grad / np.sqrt(self.cache[p_id] + self.epsilon)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"learning_rate": self.learning_rate, "decay_rate": self.decay_rate, "epsilon": self.epsilon})
        return config

    @property
    def params(self) -> dict:
        return {"cache": self.cache}


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.learning_rate: float = learning_rate
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

            for _, (param, grad) in layer.params.items():
                p_id: int = id(param)

                if p_id not in self.momentums:
                    self.momentums[p_id] = np.zeros_like(param)
                    self.velocities[p_id] = np.zeros_like(param)

                self.momentums[p_id] = self.beta1 * self.momentums[p_id] + (1 - self.beta1) * grad
                self.velocities[p_id] = self.beta2 * self.velocities[p_id] + (1 - self.beta2) * np.power(grad, 2)

                m_hat = self.momentums[p_id] / (1 - self.beta1**self.t)
                v_hat = self.velocities[p_id] / (1 - self.beta2**self.t)

                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "learning_rate": self.learning_rate,
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
