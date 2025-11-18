class Optimizer:
    def step(self, layers):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "weights_gradient") and layer.weights_gradient is not None:
                layer.weights -= self.learning_rate * layer.weights_gradient

            if hasattr(layer, "kernels_gradient") and layer.kernels_gradient is not None:
                layer.kernels -= self.learning_rate * layer.kernels_gradient

            if hasattr(layer, "biases_gradient") and layer.biases_gradient is not None:
                layer.biases -= self.learning_rate * layer.biases_gradient
