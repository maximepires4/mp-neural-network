import numpy as np
from . import utils
from .activations import Softmax, Sigmoid
from .losses import MSE, CategoricalCrossEntropy, BinaryCrossEntropy


class Model:
    def __init__(self, layers, loss, task_type=None):
        self.layers = layers
        self.loss = loss

        self.output_activation = None
        self.task_type = task_type

        if self.task_type is None:
            if isinstance(loss, MSE):
                self.task_type = "regression"
            if isinstance(loss, CategoricalCrossEntropy):
                self.output_activation = Softmax()
                self.task_type = "categorical"
            elif isinstance(loss, BinaryCrossEntropy):
                self.output_activation = Sigmoid()
                self.task_type = "binary"

    def train(self, input, output, epochs, learning_rate, batch_size):
        input_copy = np.copy(input)
        output_copy = np.copy(output)

        batches = np.floor(input_copy.shape[0] / batch_size).astype(int)

        for epoch in range(epochs):
            error = 0
            accuracy = 0 if self.task_type != "regression" else None
            count = 0
            input_copy, output_copy = utils.shuffle(input_copy, output_copy)

            for batch in range(batches):
                for layer in self.layers:
                    layer.clear_gradients()

                for x, y in zip(
                    input_copy[batch * batch_size : (batch + 1) * batch_size],
                    output_copy[batch * batch_size : (batch + 1) * batch_size],
                ):
                    y_hat = x

                    for layer in self.layers:
                        y_hat = layer.forward(y_hat)

                    error += self.loss.direct(y, y_hat)

                    if self.task_type == "categorical":
                        assert accuracy is not None
                        accuracy += 1 if np.argmax(y_hat) == np.argmax(y) else 0
                    elif self.task_type == "binary":
                        assert accuracy is not None
                        pred_label = 1 if y_hat > 0 else 0
                        accuracy += 1 if pred_label == y else 0

                    count += 1

                    grad = self.loss.prime(y, y_hat)
                    for layer in reversed(self.layers):
                        grad = layer.backward(grad)

                for layer in self.layers:
                    layer.update(learning_rate, batch_size)

                if accuracy is None:
                    msg = "epoch %d/%d   batch %d/%d   error=%f" % (
                        epoch + 1,
                        epochs,
                        batch + 1,
                        batches,
                        error / count,
                    )
                else:
                    msg = "epoch %d/%d   batch %d/%d   error=%f   accuracy=%.2f" % (
                        epoch + 1,
                        epochs,
                        batch + 1,
                        batches,
                        error / count,
                        100 * accuracy / count,
                    )

                if batch == batches - 1:
                    print(msg)
                else:
                    print(msg, end="\r")

            error /= len(input_copy)

    def test(self, input, output):
        error = 0
        accuracy = 0 if self.task_type != 'regression' else None

        for x, y in zip(input, output):
            y_hat = x
            for layer in self.layers:
                y_hat = layer.forward(y_hat)

            if self.task_type == "categorical":
                assert accuracy is not None
                accuracy += 1 if np.argmax(y_hat) == np.argmax(y) else 0
            elif self.task_type == "binary":
                assert accuracy is not None
                pred_label = 1 if y_hat > 0 else 0
                accuracy += 1 if pred_label == y else 0

            error += self.loss.direct(y, y_hat)

        len_input = len(input)
        error /= len_input

        if accuracy is not None:
            accuracy /= len_input
            print("error=%f | accuracy=%.2f" % (error, accuracy * 100))
        else:
            print("error=%f" % error)

    def predict(self, input):
        output = np.copy(input)
        for layer in self.layers:
            output = layer.forward(output)

        if self.output_activation:
            output = self.output_activation.forward(output)

        return output
