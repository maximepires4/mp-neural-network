class Optimizer:
    def step(self, layers, batch_size):
        pass

    def zero_grad(self, layers):
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def step(self, layers, batch_size):
        for layer in layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * (layer.weights_gradient / batch_size)

            if hasattr(layer, 'kernels'):
                layer.kernels -= self.learning_rate * (layer.kernels_gradient / batch_size)

            if hasattr(layer, 'biases'):
                layer.biases -= self.learning_rate * (layer.biases_gradient / batch_size)

    def zero_grad(self, layers):
        for layer in layers:
            layer.clear_gradients()
