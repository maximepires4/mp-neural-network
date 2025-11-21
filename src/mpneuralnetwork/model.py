import numpy as np
from numpy.typing import NDArray

from .activations import Activation, PReLU, ReLU, Sigmoid, Softmax, Swish
from .layers import Dense, Layer, Lit_W
from .losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy, Loss
from .optimizers import SGD, Optimizer


class Model:
    def __init__(self, layers: list[Layer], loss: Loss, optimizer: Optimizer | None = None) -> None:
        self.layers: list[Layer] = layers
        self.loss: Loss = loss
        self.optimizer: Optimizer = SGD() if optimizer is None else optimizer
        self.output_activation: Activation | Layer | None = None

        self.task_type: str

        self._init_smart_weights()
        self._init_output_activation()

        if isinstance(loss, MSE):
            self.task_type = "regression"
        else:
            self.task_type = "classification"

    def _init_smart_weights(self) -> None:
        for i in range(len(self.layers)):
            layer = self.layers[i]

            if isinstance(layer, (Dense)) and layer.initialization == "auto":
                method: Lit_W = "xavier"

                if i + 1 < len(self.layers):
                    next_layer = self.layers[i + 1]

                    if isinstance(next_layer, (ReLU, PReLU, Swish)):
                        method = "he"

                layer.init_weights(method)

    def _init_output_activation(self) -> None:
        if isinstance(self.loss, BinaryCrossEntropy):
            self.output_activation = Sigmoid()

        elif isinstance(self.loss, CategoricalCrossEntropy):
            self.output_activation = Softmax()

        if not self.output_activation:
            return

        if isinstance(self.layers[len(self.layers) - 1], type(self.output_activation)):
            self.layers = self.layers[:-1]

    def train(self, X_train: NDArray, y_train: NDArray, epochs: int, batch_size: int) -> None:
        num_samples: int = X_train.shape[0]
        num_batches: int = int(np.floor(num_samples / batch_size))

        for epoch in range(epochs):
            error: float = 0
            accuracy: float | None = 0 if self.task_type != "regression" else None

            permutation: NDArray = np.random.permutation(num_samples)
            X_shuffled: NDArray = X_train[permutation]
            y_shuffled: NDArray = y_train[permutation]

            for i in range(num_batches):
                start: int = i * batch_size
                end: int = (i + 1) * batch_size
                X_batch: NDArray = X_shuffled[start:end]
                y_batch: NDArray = y_shuffled[start:end]

                predictions: NDArray = X_batch
                for layer in self.layers:
                    predictions = layer.forward(predictions)

                error += self.loss.direct(predictions, y_batch)

                if accuracy is not None:
                    if y_batch.ndim == 2 and y_batch.shape[1] > 1:
                        correct_predictions = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1))
                        accuracy += correct_predictions
                    elif y_batch.ndim == 1:
                        pred_labels = (predictions > 0).astype(int)
                        accuracy += np.sum(pred_labels == y_batch)

                grad: NDArray = self.loss.prime(predictions, y_batch)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                self.optimizer.step(self.layers)

            error /= num_batches

            if accuracy is not None:
                total_samples = num_batches * batch_size
                accuracy /= total_samples
                print(f"Epoch {epoch + 1}/{epochs}      error={error:.4f}       accuracy={accuracy * 100:.2f}%")
            else:
                print(f"Epoch {epoch + 1}/{epochs}      error={error:.4f}")

    def test(self, X_test: NDArray, y_test: NDArray) -> None:
        logits: NDArray = np.copy(X_test)
        for layer in self.layers:
            logits = layer.forward(logits, training=False)

        error: float = self.loss.direct(logits, y_test)

        predictions: NDArray = logits
        if self.output_activation is not None:
            predictions = self.output_activation.forward(logits)

        accuracy: float | None = None
        if self.task_type != "regression":
            if y_test.ndim == 2 and y_test.shape[1] > 1:
                accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
            elif y_test.ndim == 1:
                pred_labels = (predictions > 0.5).astype(int)
                accuracy = np.sum(pred_labels == y_test) / y_test.shape[0]

        if accuracy is not None:
            print(f"Test Error: {error:.4f}, Test Accuracy: {accuracy * 100:.2f}%")
        else:
            print(f"Test Error: {error:.4f}")

    def predict(self, input: NDArray) -> NDArray:
        output: NDArray = np.copy(input)
        for layer in self.layers:
            output = layer.forward(output, training=False)

        if self.output_activation is not None:
            output = self.output_activation.forward(output)

        return output
