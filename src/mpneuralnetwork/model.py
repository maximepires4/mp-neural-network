import numpy as np

from .activations import Sigmoid, Softmax
from .losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from .optimizers import SGD


class Model:
    def __init__(self, layers, loss, optimizer=SGD()):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

        if isinstance(loss, MSE):
            self.task_type = "regression"
        else:
            self.task_type = "classification"

    def train(self, X_train, y_train, epochs, batch_size):
        num_samples = X_train.shape[0]
        num_batches = int(np.floor(num_samples / batch_size))

        for epoch in range(epochs):
            error = 0
            accuracy = 0 if self.task_type != "regression" else None

            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                predictions = X_batch
                for layer in self.layers:
                    predictions = layer.forward(predictions)

                error += self.loss.direct(predictions, y_batch)

                if accuracy is not None:
                    if y_batch.ndim == 2 and y_batch.shape[1] > 1:
                        correct_predictions = np.sum(
                            np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1)
                        )
                        accuracy += correct_predictions
                    elif y_batch.ndim == 1:
                        pred_labels = (predictions > 0).astype(int)
                        accuracy += np.sum(pred_labels == y_batch)

                grad = self.loss.prime(predictions, y_batch)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                self.optimizer.step(self.layers)

            error /= num_batches

            if accuracy is not None:
                total_samples = num_batches * batch_size
                accuracy /= total_samples
                print(f"Epoch {epoch + 1}/{epochs}      error={error:.4f}       accuracy={accuracy*100:.2f}%")
            else:
                print(f"Epoch {epoch + 1}/{epochs}      error={error:.4f}")

    def test(self, X_test, y_test):
        predictions = self.predict(X_test)
        error = self.loss.direct(predictions, y_test)

        accuracy = None
        if self.task_type != "regression":
            if y_test.ndim == 2 and y_test.shape[1] > 1:
                accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
            elif y_test.ndim == 1:
                pred_labels = (predictions > 0.5).astype(int)
                accuracy = np.sum(pred_labels == y_test) / y_test.shape[0]

        if accuracy is not None:
            print(f"Test Error: {error:.4f}, Test Accuracy: {accuracy*100:.2f}%")
        else:
            print(f"Test Error: {error:.4f}")

    def predict(self, input):
        output = np.copy(input)
        for layer in self.layers:
            output = layer.forward(output, training=False)
        
        if isinstance(self.loss, CategoricalCrossEntropy):
            output = Softmax().forward(output)
        elif isinstance(self.loss, BinaryCrossEntropy):
            output = Sigmoid().forward(output)

        return output
