import numpy as np
from numpy.typing import NDArray

from mpneuralnetwork.metrics import RMSE, Accuracy, F1Score, Metric, R2Score

from .activations import Activation, PReLU, ReLU, Sigmoid, Softmax, Swish
from .layers import BatchNormalization, Convolutional, Dense, Dropout, Layer, Lit_W
from .losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy, Loss
from .optimizers import SGD, Optimizer


class Model:
    def __init__(self, layers: list[Layer], loss: Loss, optimizer: Optimizer | None = None, metrics: list[Metric] | None = None) -> None:
        self.layers: list[Layer] = layers
        self.loss: Loss = loss
        self.optimizer: Optimizer = SGD() if optimizer is None else optimizer
        self.metrics: list[Metric] = metrics if metrics is not None else []
        self.output_activation: Activation | Layer | None = None

        self._build_graph()
        self._init_smart_weights()
        self._init_output_activation()
        self._init_smart_metrics()

    def _build_graph(self) -> None:
        first_layer = self.layers[0]

        if not hasattr(first_layer, "input_size") or first_layer.input_size is None:
            raise ValueError("Input layer does not define input size")

        current_output_size: int = first_layer.output_size

        for i in range(1, len(self.layers)):
            layer = self.layers[i]

            layer.build(current_output_size)

            if hasattr(layer, "output_size"):
                current_output_size = layer.output_size

    def _init_smart_weights(self) -> None:
        for i in range(len(self.layers)):
            layer = self.layers[i]

            if isinstance(layer, (Dense, Convolutional)) and layer.initialization == "auto":
                method: Lit_W = "xavier"
                no_bias: bool = False

                for j in range(i + 1, len(self.layers)):
                    next_layer = self.layers[j]

                    if isinstance(next_layer, BatchNormalization):
                        no_bias = True
                        continue

                    if isinstance(next_layer, Dropout):
                        continue

                    if isinstance(next_layer, (ReLU, PReLU, Swish)):
                        method = "he"
                        break

                    if isinstance(next_layer, (Activation, Dense, Convolutional)):
                        break

                layer.init_weights(method, no_bias)

    def _init_output_activation(self) -> None:
        if isinstance(self.loss, BinaryCrossEntropy):
            self.output_activation = Sigmoid()

        elif isinstance(self.loss, CategoricalCrossEntropy):
            self.output_activation = Softmax()

        if not self.output_activation:
            return

        if isinstance(self.layers[len(self.layers) - 1], type(self.output_activation)):
            self.layers = self.layers[:-1]

    def _init_smart_metrics(self) -> None:
        if len(self.metrics) != 0:
            return

        if isinstance(self.loss, MSE):
            self.metrics = [RMSE(), R2Score()]
        else:
            self.metrics = [Accuracy(), F1Score()]

    def train(
        self,
        X_train: NDArray,
        y_train: NDArray,
        epochs: int,
        batch_size: int,
        evaluation: tuple[NDArray, NDArray] | None = None,
        auto_evaluation: float = 0.2,
        early_stopping: int | None = None,
        model_checkpoint: bool = True,
    ) -> None:
        X_copy: NDArray = np.copy(X_train)
        y_copy: NDArray = np.copy(y_train)

        permutation: NDArray
        if evaluation is None and auto_evaluation != 0:
            split_i = int(len(X_train) * auto_evaluation)

            permutation = np.random.permutation(X_copy.shape[0])
            X_copy = X_copy[permutation]
            y_copy = y_copy[permutation]

            X_copy, y_copy = X_train[:-split_i], y_train[:-split_i]
            evaluation = (X_train[-split_i:], y_train[-split_i:])

        num_samples = X_copy.shape[0]
        num_batches = int(np.floor(num_samples / batch_size))

        early_stopping = early_stopping if early_stopping else epochs + 1
        patience: int = early_stopping
        best_error: float = float("inf")
        best_weights: dict | None = None

        for epoch in range(epochs):
            metric_dict: dict[str, float] = {}
            metric_dict["loss"] = 0

            for metric in self.metrics:
                metric_dict[metric.__class__.__name__.lower()] = 0

            permutation = np.random.permutation(num_samples)
            X_shuffled = X_copy[permutation]
            y_shuffled = y_copy[permutation]

            for i in range(num_batches):
                start: int = i * batch_size
                end: int = (i + 1) * batch_size
                X_batch: NDArray = X_shuffled[start:end]
                y_batch: NDArray = y_shuffled[start:end]

                predictions, new_metric_dict = self.evaluate(X_batch, y_batch, training=True)

                for key, value in new_metric_dict.items():
                    metric_dict[key] += value

                grad: NDArray = self.loss.prime(predictions, y_batch)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                self.optimizer.step(self.layers)

            spacing_str = " " * abs(len(str(epochs)) - len(str(epoch + 1)))
            message = f"epoch {spacing_str}{epoch + 1}/{epochs}   |   [training]"

            for key, _ in metric_dict.items():
                metric_dict[key] /= num_batches
                message += f"   {key} = {metric_dict[key]:.4f}"

            if evaluation is not None:
                _, val_metric_dict = self.evaluate(evaluation[0], evaluation[1], training=False)

                if val_metric_dict["loss"] < best_error:
                    best_error = val_metric_dict["loss"]
                    patience = early_stopping
                    if model_checkpoint:
                        best_weights = self.get_weights()
                else:
                    patience -= 1

                message += "   |   [evaluation]"
                for key, value in val_metric_dict.items():
                    message += f"   {key} = {value:.4f}"

            elif metric_dict["loss"] < best_error:
                best_error = metric_dict["loss"]
                patience = early_stopping
                if model_checkpoint:
                    best_weights = self.get_weights()
            else:
                patience -= 1

            print(message)

            if patience == 0:
                print(f"EARLY STOPPING - Model did not learn since {early_stopping} epochs")
                break

        if model_checkpoint and best_weights is not None:
            self.restore_weights(best_weights)
            print(f"MODEL CHECKPOINT: {best_error:.4f}")
            # TODO: Save also optimizer state, better user output

    def evaluate(self, X: NDArray, y: NDArray, training=False) -> tuple[NDArray, dict[str, float]]:
        logits: NDArray = np.copy(X)
        for layer in self.layers:
            logits = layer.forward(logits, training=training)

        loss: float = self.loss.direct(logits, y)

        metric_dict: dict[str, float] = {}
        metric_dict["loss"] = loss

        predictions_metrics: NDArray = logits
        if self.output_activation is not None:
            predictions_metrics = self.output_activation.forward(logits)

        for metric in self.metrics:
            key = metric.__class__.__name__.lower()

            if isinstance(metric, RMSE) and isinstance(self.loss, MSE):
                metric_dict[key] = metric.from_mse(loss)
            else:
                metric_dict[key] = metric(y, predictions_metrics)

        predictions: NDArray = logits
        if not training:
            predictions = predictions_metrics

        return predictions, metric_dict

    def test(self, X_test: NDArray, y_test: NDArray) -> None:
        _, metric_dict = self.evaluate(X_test, y_test, training=False)

        print("Test resuls:")
        for key, value in metric_dict.items():
            print(f"   {key} = {value:.4f}")

    def predict(self, input: NDArray) -> NDArray:
        output: NDArray = np.copy(input)
        for layer in self.layers:
            output = layer.forward(output, training=False)

        if self.output_activation is not None:
            output = self.output_activation.forward(output)

        return output

    def get_weights(self, optimizer_params: dict | None = None) -> dict:
        weights_dict: dict = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "params"):
                for l_param_name, (l_param, _) in layer.params.items():
                    p_id: int = id(l_param)
                    logical_name: str = f"layer_{i}_{l_param_name}"

                    weights_dict[logical_name] = np.copy(l_param)

                    if not optimizer_params:
                        continue

                    for o_param_name, o_param in optimizer_params.items():
                        if p_id in o_param:
                            weights_dict[f"optimizer_{o_param_name}_{logical_name}"] = np.copy(o_param[p_id])

            if hasattr(layer, "state"):
                for l_state_name, l_state_val in layer.state.items():
                    logical_name = f"layer_{i}_state_{l_state_name}"
                    weights_dict[logical_name] = np.copy(l_state_val)

        return weights_dict

    def restore_weights(self, weights_dict: dict, optimizer: Optimizer | None = None) -> None:
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "params"):
                current_params = {}
                for l_param_name in layer.params:
                    logical_name = f"layer_{i}_{l_param_name}"
                    if logical_name in weights_dict:
                        current_params[l_param_name] = weights_dict[logical_name]

                if hasattr(layer, "load_params") and current_params:
                    layer.load_params(current_params)

                if optimizer:
                    for l_param_name, (l_param, _) in layer.params.items():
                        p_id = id(l_param)
                        logical_name = f"layer_{i}_{l_param_name}"

                        for o_param_name, o_param in optimizer.params.items():
                            if not isinstance(o_param, dict):
                                continue
                            key = f"optimizer_{o_param_name}_{logical_name}"
                            if key in weights_dict:
                                if p_id not in o_param:
                                    o_param[p_id] = np.zeros_like(l_param)
                                np.copyto(o_param[p_id], weights_dict[key])

            if hasattr(layer, "state"):
                current_state = {}
                for l_state_name in layer.state:
                    logical_name = f"layer_{i}_state_{l_state_name}"
                    if logical_name in weights_dict:
                        current_state[l_state_name] = weights_dict[logical_name]

                if current_state:
                    layer.state = current_state
