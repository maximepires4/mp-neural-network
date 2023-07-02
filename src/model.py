import random
import numpy as np
from . import utils


class Model:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def train(self, input, output, epochs, learning_rate):

        input_copy = np.copy(input)
        output_copy = np.copy(output)

        for epoch in range(epochs):
            error = 0
            input_copy, output_copy = utils.shuffle(input_copy, output_copy)
            
            for x, y in zip(input_copy, output_copy):
                y_hat = x

                for layer in self.layers:
                    y_hat = layer.forward(y_hat)

                error += self.loss.direct(y, y_hat)

                grad = self.loss.prime(y, y_hat)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(input_copy)
            print('epoch %d/%d   error=%f' % (epoch+1, epochs, error))

    def test(self, input, output):
        error = 0
        for x, y in zip(input, output):
            y_hat = x
            for layer in self.layers:
                y_hat = layer.forward(y_hat)

            error += self.loss.direct(y, y_hat)

        error /= len(input)
        print('error=%f' % error)

    def predict(self, input):
        output = np.copy(input)
        for layer in self.layers:
            output = layer.forward(output)
        return output