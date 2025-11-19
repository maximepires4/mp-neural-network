import numpy as np


class Optimizer:
    def step(self, layers):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            if (
                hasattr(layer, "weights_gradient")
                and layer.weights_gradient is not None
            ):
                layer.weights -= self.learning_rate * layer.weights_gradient

            if (
                hasattr(layer, "kernels_gradient")
                and layer.kernels_gradient is not None
            ):
                layer.kernels -= self.learning_rate * layer.kernels_gradient

            if hasattr(layer, "biases_gradient") and layer.biases_gradient is not None:
                layer.biases -= self.learning_rate * layer.biases_gradient


class SGDMomentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.1):
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocities_w = {}
        self.velocities_b = {}
        self.velocities_k = {}

    def step(self, layers):
        for layer in layers:
            if (
                hasattr(layer, "weights_gradient")
                and layer.weights_gradient is not None
            ):
                if layer not in self.velocities_w:
                    self.velocities_w[layer] = np.zeros_like(layer.weights)

                velocity = (
                    self.momentum * self.velocities_w[layer]
                    - self.learning_rate * layer.weights_gradient
                )

                self.velocities_w[layer] = velocity
                layer.weights += velocity

            if (
                hasattr(layer, "kernels_gradient")
                and layer.kernels_gradient is not None
            ):
                if layer not in self.velocities_k:
                    self.velocities_k[layer] = np.zeros_like(layer.kernels)

                velocity = (
                    self.momentum * self.velocities_k[layer]
                    - self.learning_rate * layer.kernels_gradient
                )

                self.velocities_k[layer] = velocity
                layer.kernels += velocity

            if hasattr(layer, "biases_gradient") and layer.biases_gradient is not None:
                if layer not in self.velocities_b:
                    self.velocities_b[layer] = np.zeros_like(layer.biases)

                velocity = (
                    self.momentum * self.velocities_b[layer]
                    - self.learning_rate * layer.biases_gradient
                )

                self.velocities_b[layer] = velocity
                layer.biases += velocity


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon

        self.cache_w = {}
        self.cache_b = {}
        self.cache_k = {}

    def step(self, layers):
        for layer in layers:
            if (
                hasattr(layer, "weights_gradient")
                and layer.weights_gradient is not None
            ):
                if layer not in self.cache_w:
                    self.cache_w[layer] = np.zeros_like(layer.weights)

                self.cache_w[layer] = self.decay_rate * self.cache_w[layer] + (
                    1 - self.decay_rate
                ) * np.power(layer.weights_gradient, 2)

                layer.weights -= (
                    self.learning_rate
                    * layer.weights_gradient
                    / np.sqrt(self.cache_w[layer] + self.epsilon)
                )

            if (
                hasattr(layer, "kernels_gradient")
                and layer.kernels_gradient is not None
            ):
                if layer not in self.cache_k:
                    self.cache_k[layer] = np.zeros_like(layer.kernels)

                self.cache_k[layer] = self.decay_rate * self.cache_k[layer] + (
                    1 - self.decay_rate
                ) * np.power(layer.kernels_gradient, 2)

                layer.kernels -= (
                    self.learning_rate
                    * layer.kernels_gradient
                    / np.sqrt(self.cache_k[layer] + self.epsilon)
                )

            if hasattr(layer, "biases_gradient") and layer.biases_gradient is not None:
                if layer not in self.cache_b:
                    self.cache_b[layer] = np.zeros_like(layer.biases)

                self.cache_b[layer] = self.decay_rate * self.cache_b[layer] + (
                    1 - self.decay_rate
                ) * np.power(layer.biases_gradient, 2)

                layer.biases -= (
                    self.learning_rate
                    * layer.biases_gradient
                    / np.sqrt(self.cache_b[layer] + self.epsilon)
                )


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m_w, self.m_b, self.m_k = {}, {}, {}
        self.v_w, self.v_b, self.v_k = {}, {}, {}

    def step(self, layers):
        self.t += 1
        for layer in layers:

            def apply_adam(grad, m_cache, v_cache):
                if layer not in m_cache:
                    m_cache[layer] = np.zeros_like(grad)
                    v_cache[layer] = np.zeros_like(grad)
        
                m_cache[layer] = self.beta1 * m_cache[layer] + (1 - self.beta1) * grad
                v_cache[layer] = self.beta2 * v_cache[layer] + (1 - self.beta2) * np.power(grad, 2)

                m_hat = m_cache[layer] / (1 - self.beta1 ** self.t)
                v_hat = v_cache[layer] / (1 - self.beta2 ** self.t)
        
                return self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            if (
                hasattr(layer, "weights_gradient")
                and layer.weights_gradient is not None
            ):
                layer.weights -= apply_adam(layer.weights_gradient, self.m_w, self.v_w)

            if (
                hasattr(layer, "kernels_gradient")
                and layer.kernels_gradient is not None
            ):
                layer.kernels -= apply_adam(layer.kernels_gradient, self.m_k, self.v_k)

            if hasattr(layer, "biases_gradient") and layer.biases_gradient is not None:
                layer.biases -= apply_adam(layer.biases_gradient, self.m_b, self.v_b)
