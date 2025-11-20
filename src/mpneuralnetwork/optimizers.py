import numpy as np


class Optimizer:
    def step(self, layers):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.1):
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocities = {}

    def step(self, layers):
        for layer in layers:

            if not hasattr(layer, 'params'):
                continue

            for _, (param, grad) in layer.params.items():
                p_id = id(param)

                if p_id not in self.velocities:
                    self.velocities[p_id] = np.zeros_like(param)

                self.velocities[p_id] = self.momentum * self.velocities[p_id] - self.learning_rate * grad
                param += self.velocities[p_id]


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon

        self.cache = {}

    def step(self, layers):
        for layer in layers:

            if not hasattr(layer, 'params'):
                continue

            for _, (param, grad) in layer.params.items():
                p_id = id(param)

                if p_id not in self.cache:
                    self.cache[p_id] = np.zeros_like(param)

                self.cache[p_id] = self.decay_rate * self.cache[p_id] + (1 - self.decay_rate) * np.power(grad, 2)

                param -= self.learning_rate * grad / np.sqrt(self.cache[p_id] + self.epsilon)


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0
        self.momentums = {}
        self.velocities = {}

    def step(self, layers):
        self.t += 1

        for layer in layers:

            if not hasattr(layer, 'params'):
                continue

            for param_name, (param, grad) in layer.params.items():
                p_id = id(param)

                if p_id not in self.momentums:
                    self.momentums[p_id] = np.zeros_like(param)
                    self.velocities[p_id] = np.zeros_like(param)

                self.momentums[p_id] = self.beta1 * self.momentums[p_id] + (1 - self.beta1) * grad
                self.velocities[p_id] = self.beta2 * self.velocities[p_id] + (1 - self.beta2) * np.power(grad, 2)

                m_hat = self.momentums[p_id] / (1 - self.beta1 ** self.t)
                v_hat = self.velocities[p_id] / (1 - self.beta2 ** self.t)

                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
