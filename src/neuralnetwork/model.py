import random
import numpy as np
from . import utils


class Model:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def train(self, input, output, epochs, learning_rate, batch_size):

        input_copy = np.copy(input)
        output_copy = np.copy(output)

        batches = np.floor(input_copy.shape[0] / batch_size).astype(int)

        for epoch in range(epochs):
            error = 0
            input_copy, output_copy = utils.shuffle(input_copy, output_copy)

            for batch in range(batches):

                # print('batch %d/%d, %d' % (batch+1, np.floor(input_copy.shape[0] / batch_size).astype(int), len(input_copy[batch*batch_size:(batch+1)*batch_size])))

                for layer in self.layers:
                    layer.clear_gradients()

                for x, y in zip(input_copy[batch*batch_size:(batch+1)*batch_size], output_copy[batch*batch_size:(batch+1)*batch_size]):

                    # for x, y in zip(input_copy, output_copy):

                    # for layer in self.layers:
                    #    layer.clear_gradients()

                    y_hat = x

                    for layer in self.layers:
                        y_hat = layer.forward(y_hat)

                    error += self.loss.direct(y, y_hat)

                    grad = self.loss.prime(y, y_hat)
                    for layer in reversed(self.layers):
                        grad = layer.backward(grad)

                    # for layer in self.layers:
                    #    layer.update(learning_rate, 1)

                for layer in self.layers:
                    layer.update(learning_rate, batch_size)

                msg = 'epoch %d/%d   batch %d/%d   error=%f' % (epoch + 1, epochs, batch+1, batches, error / len(input_copy[batch*batch_size:(batch+1)*batch_size]))

                if batch == batches - 1:
                    print(msg)
                else:
                    print(msg, end='\r')

            error /= len(input_copy)
            #print('epoch %d/%d   error=%f' % (epoch+1, epochs, error))

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
