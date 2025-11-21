import json
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .activations import Activation, PReLU, ReLU, Sigmoid, Softmax, Swish, Tanh
from .layers import Convolutional, Dense, Dropout, Layer, Lit_W, Reshape
from .losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy, Loss
from .optimizers import SGD, Adam, Optimizer, RMSprop


class Model:
    def __init__(self, layers: list[Layer], loss: Loss, optimizer: Optimizer | None = None) -> None:
        self.layers: list[Layer] = layers
        self.loss: Loss = loss
        self.optimizer: Optimizer = SGD() if optimizer is None else optimizer
        self.output_activation: Activation | Layer | None = None

        self.task_type: str

        self._build_graph()
        self._init_smart_weights()
        self._init_output_activation()

        if isinstance(loss, MSE):
            self.task_type = "regression"
        else:
            self.task_type = "classification"

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

    def train(self, X_train: NDArray, y_train: NDArray, epochs: int, batch_size: int, evaluation: tuple[NDArray, NDArray]) -> None:
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

                predictions, (new_error, new_accuracy) = self.evaluate(X_batch, y_batch, training=True)
                error += new_error
                if accuracy is not None and new_accuracy is not None:
                    accuracy += new_accuracy

                grad: NDArray = self.loss.prime(predictions, y_batch)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                self.optimizer.step(self.layers)

            error /= num_batches

            _, (val_error, val_accuracy) = self.evaluate(evaluation[0], evaluation[1], training=False)

            if accuracy is not None and val_accuracy is not None:
                accuracy /= num_batches
                print(
                    f"epoch {epoch + 1}/{epochs}        error={error:.4f}       accuracy={accuracy * 100:.2f}%      |"
                    + f"        val_error={val_error:.4f}       val_accuracy={val_accuracy * 100:.2f}%"
                )
            else:
                print(f"epoch {epoch + 1}/{epochs}      error={error:.4f}       |       val_error={val_error:.4f}")

    def evaluate(self, X: NDArray, y: NDArray, training=False) -> tuple[NDArray, tuple[float, float | None]]:
        logits: NDArray = np.copy(X)
        for layer in self.layers:
            logits = layer.forward(logits, training=training)

        error: float = self.loss.direct(logits, y)

        predictions: NDArray = logits
        if not training and self.output_activation is not None:
            predictions = self.output_activation.forward(logits)

        accuracy: float | None = None
        if self.task_type != "regression":
            if y.ndim == 2 and y.shape[1] > 1:
                accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / y.shape[0]
            elif y.ndim == 1:
                pred_labels = (predictions > 0.5).astype(int)
                accuracy = np.sum(pred_labels == y) / y.shape[0]

        return predictions, (error, accuracy)

    def test(self, X_test: NDArray, y_test: NDArray) -> None:
        _, (error, accuracy) = self.evaluate(X_test, y_test, training=False)

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

    def save(self, filepath: str) -> None:
        layers_config: list = []
        for layer in self.layers:
            layers_config.append(layer.get_config())

        optimizer_globals: dict = {}
        optimizer_params: dict = {}

        if self.optimizer.params:
            for param_name, param in self.optimizer.params.items():
                if isinstance(param, (int, float, str, bool)) or param is None:
                    optimizer_globals[param_name] = param
                else:
                    optimizer_params[param_name] = param

        model_config: dict = {
            "loss": self.loss.get_config(),
            "optimizer": self.optimizer.get_config(),
            "optimizer_globals": optimizer_globals,
        }

        weights_dict: dict = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "params"):
                for l_param_name, (l_param, _) in layer.params.items():
                    p_id: int = id(l_param)
                    logical_name: str = f"layer_{i}_{l_param_name}"

                    weights_dict[logical_name] = l_param

                    for o_param_name, o_param in optimizer_params.items():
                        if p_id in o_param:
                            weights_dict[f"optimizer_{o_param_name}_{logical_name}"] = o_param[p_id]

        if filepath[-4:] != ".npz":
            filepath = f"{filepath}.npz"

        save_data: dict = weights_dict.copy()
        save_data["architecture"] = json.dumps(layers_config)
        save_data["model_config"] = json.dumps(model_config)

        np.savez_compressed(filepath, **save_data)

    REGISTRY = {
        # Layers
        "Layer": Layer,
        "Dense": Dense,
        "Dropout": Dropout,
        "Convolutional": Convolutional,
        "Reshape": Reshape,
        # Activations
        "Activation": Activation,
        "PReLU": PReLU,
        "ReLU": ReLU,
        "Sigmoid": Sigmoid,
        "Softmax": Softmax,
        "Swish": Swish,
        "Tanh": Tanh,
        # Losses
        "Loss": Loss,
        "MSE": MSE,
        "BinaryCrossEntropy": BinaryCrossEntropy,
        "CategoricalCrossEntropy": CategoricalCrossEntropy,
        # Optimizers
        "Optimizer": Optimizer,
        "SGD": SGD,
        "Adam": Adam,
        "RMSprop": RMSprop,
    }

    @staticmethod
    def load(filepath: str) -> "Model":
        if filepath[-4:] != ".npz":
            filepath = f"{filepath}.npz"

        data: dict = np.load(filepath, allow_pickle=True)

        layers_config: dict = json.loads(str(data["architecture"]))
        model_config: dict = json.loads(str(data["model_config"]))

        layers: list[Layer] = []
        for conf in layers_config:
            layer_type: str = str(conf.pop("type"))
            layer_class: Callable | None = Model.REGISTRY.get(layer_type)

            if layer_class is None:
                raise ValueError(f"Unknown layer type: {layer_type}")

            layer: Layer = layer_class(**conf)
            layers.append(layer)

        loss_conf: dict = model_config["loss"]
        loss_type: str = str(loss_conf.pop("type"))
        loss_class: Callable | None = Model.REGISTRY.get(loss_type)

        if loss_class is None:
            raise ValueError(f"Unknown loss type: {loss_type}")

        loss: Loss = loss_class(**loss_conf)

        optim_conf: dict | str = model_config["optimizer"]
        optim_type: str

        if isinstance(optim_conf, dict):
            optim_type = str(optim_conf.pop("type"))
        else:
            optim_type = str(optim_conf)
            optim_conf = {}

        optim_class: Callable | None = Model.REGISTRY.get(optim_type)

        if optim_class is None:
            raise ValueError(f"Unknown optimizer type: {optim_type}")

        optimizer: Optimizer = optim_class(**optim_conf)

        if "optimizer_globals" in model_config:
            for param_name, param in model_config["optimizer_globals"].items():
                setattr(optimizer, param_name, param)

        model = Model(layers, loss, optimizer)

        for i, layer in enumerate(model.layers):
            if hasattr(layer, "params"):
                for l_param_name, (_, _) in layer.params.items():
                    logical_name = f"layer_{i}_{l_param_name}"

                    if logical_name in data:
                        setattr(layer, l_param_name, data[logical_name])

                        new_param_val = getattr(layer, l_param_name)
                        p_id = id(new_param_val)

                        for o_param_name, o_param in optimizer.params.items():
                            if not isinstance(o_param, dict):
                                continue

                            key = f"optimizer_{o_param_name}_{logical_name}"
                            if key in data:
                                o_param[p_id] = data[key]

        return model
