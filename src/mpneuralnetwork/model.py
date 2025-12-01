import numpy as np

from . import DTYPE, ArrayType, to_device, to_host, xp
from .activations import Activation, PReLU, ReLU, Sigmoid, Softmax, Swish
from .layers import BatchNormalization, Convolutional, Dense, Dropout, Layer, Lit_W
from .losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy, Loss
from .metrics import RMSE, Accuracy, F1Score, Metric, R2Score
from .optimizers import SGD, Adam, Optimizer
from .serialization import get_model_weights, restore_model_weights


class Model:
    """The main container for building and training neural networks.

    This class handles the assembly of layers, the training loop, validation,
    and prediction. It also implements "smart features" like automatic weight
    initialization and metric selection.

    Attributes:
        layers (list[Layer]): List of layers in the network.
        loss (Loss): The loss function to minimize.
        optimizer (Optimizer): The optimization algorithm.
        metrics (list[Metric]): List of metrics to monitor.
    """

    def __init__(
        self,
        layers: list[Layer],
        loss: Loss,
        optimizer: Optimizer | None = None,
        metrics: list[Metric] | None = None,
    ) -> None:
        """Initializes the Model.

        Args:
            layers (list[Layer]): The sequence of layers.
            loss (Loss): The objective function.
            optimizer (Optimizer | None, optional): Optimizer instance. Defaults to SGD().
            metrics (list[Metric] | None, optional): Metrics to track. Defaults to [].
        """
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
        """Builds the computational graph by connecting layers.

        Propagates shape information from the first layer through the rest of the network.
        """
        first_layer = self.layers[0]

        if not hasattr(first_layer, "input_size") or first_layer.input_size is None:
            raise ValueError("Input layer does not define input size")

        current_output_size: tuple[int, ...] = first_layer.output_shape

        for i in range(1, len(self.layers)):
            layer = self.layers[i]

            layer.build(current_output_size)

            if hasattr(layer, "output_shape"):
                current_output_size = layer.output_shape

    def _init_smart_weights(self) -> None:
        """Automatically initializes weights based on activation functions.

        Uses He initialization for ReLU-like activations and Xavier for Sigmoid/Tanh.
        Also handles bias disabling for BatchNormalization.
        """
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
        """Configures the final activation for numerical stability.

        The framework uses logits for Loss functions. This method ensures the
        user hasn't redundantly added a Softmax/Sigmoid layer at the end if the
        loss function expects logits, and sets up the implicit output activation
        for predictions.
        """
        if isinstance(self.loss, BinaryCrossEntropy):
            self.output_activation = Sigmoid()

        elif isinstance(self.loss, CategoricalCrossEntropy):
            self.output_activation = Softmax()

        if not self.output_activation:
            return

        if isinstance(self.layers[len(self.layers) - 1], type(self.output_activation)):
            self.layers = self.layers[:-1]

    def _init_smart_metrics(self) -> None:
        """Automatically selects default metrics if none are provided.

        - **Regression (MSE Loss):** Defaults to [RMSE, R2Score].
        - **Classification (CrossEntropy):** Defaults to [Accuracy, F1Score].
        """
        if len(self.metrics) != 0:
            return

        if isinstance(self.loss, MSE):
            self.metrics = [RMSE(), R2Score()]
        else:
            self.metrics = [Accuracy(), F1Score()]

    def train(
        self,
        X_train: ArrayType,
        y_train: ArrayType,
        epochs: int,
        batch_size: int,
        evaluation: tuple[ArrayType, ArrayType] | None = None,
        auto_evaluation: float = 0.2,
        early_stopping: int | None = None,
        model_checkpoint: bool = True,
        compute_train_metrics: bool = False,
    ) -> None:
        """Trains the model using the provided data.

        Args:
            X_train (ArrayType): Training features.
            y_train (ArrayType): Training labels (one-hot encoded for classification).
            epochs (int): Number of complete passes through the dataset.
            batch_size (int): Number of samples per gradient update.
            evaluation (tuple[ArrayType, ArrayType] | None, optional): Explicit validation set (X_val, y_val).
            auto_evaluation (float, optional): Fraction of training data to use for validation if 'evaluation' is None.
            early_stopping (int | None, optional): Number of epochs with no improvement to wait before stopping.
            model_checkpoint (bool, optional): Whether to restore the best weights after training. Defaults to True.
            compute_train_metrics (bool, optional): Whether to compute expensive metrics on training data every epoch.
        """
        X_t = to_device(X_train.astype(DTYPE, copy=False))
        y_t = to_device(y_train.astype(DTYPE, copy=False))

        X_val: ArrayType | None = None
        y_val: ArrayType | None = None

        if evaluation is not None:
            X_val = to_device(evaluation[0].astype(DTYPE, copy=False))
            y_val = to_device(evaluation[1].astype(DTYPE, copy=False))

        elif auto_evaluation > 0.0:
            split_i = int(len(X_t) * auto_evaluation)

            all_indices = xp.random.permutation(X_t.shape[0])

            train_indices = all_indices[:-split_i]
            val_indices = all_indices[-split_i:]

            X_val = X_t[val_indices]
            y_val = y_t[val_indices]

            X_t = X_t[train_indices]
            y_t = y_t[train_indices]

        num_samples = X_t.shape[0]
        num_batches = int(np.floor(num_samples / batch_size))

        early_stopping = early_stopping if early_stopping else epochs + 1
        patience: int = early_stopping
        best_error: float = float("inf")
        best_weights: dict | None = None
        temp_t: int = 0  # TODO: Find a better solution

        for epoch in range(epochs):
            metric_dict: dict[str, float] = {}
            metric_dict["loss"] = 0

            if compute_train_metrics:
                for metric in self.metrics:
                    metric_dict[metric.__class__.__name__.lower()] = 0

            indices = xp.arange(num_samples)
            xp.random.shuffle(indices)

            for i in range(num_batches):
                batch_idx = indices[i * batch_size : (i + 1) * batch_size]
                X_batch: ArrayType = X_t[batch_idx]
                y_batch: ArrayType = y_t[batch_idx]

                predictions, new_metric_dict = self.evaluate(
                    X_batch,
                    y_batch,
                    training=True,
                    compute_metrics=compute_train_metrics,
                )

                for key, value in new_metric_dict.items():
                    metric_dict[key] += value

                grad: ArrayType = self.loss.prime(predictions, y_batch)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                self.optimizer.step(self.layers)

            spacing_str = " " * abs(len(str(epochs)) - len(str(epoch + 1)))
            message = f"epoch {spacing_str}{epoch + 1}/{epochs}   |   [training]"

            for key, _ in metric_dict.items():
                metric_dict[key] /= num_batches
                message += f"   {key} = {metric_dict[key]:.4f}"

            if X_val is not None and y_val is not None:
                _, val_metric_dict = self.evaluate(X_val, y_val, training=False)

                if val_metric_dict["loss"] < best_error:
                    best_error = val_metric_dict["loss"]
                    patience = early_stopping
                    if model_checkpoint:
                        best_weights = get_model_weights(self.layers)
                        if isinstance(self.optimizer, Adam):
                            temp_t = self.optimizer.t
                else:
                    patience -= 1

                message += "   |   [evaluation]"
                for key, value in val_metric_dict.items():
                    message += f"   {key} = {value:.4f}"

            elif metric_dict["loss"] < best_error:
                best_error = metric_dict["loss"]
                patience = early_stopping
                if model_checkpoint:
                    best_weights = get_model_weights(self.layers)
                    if isinstance(self.optimizer, Adam):
                        temp_t = self.optimizer.t
            else:
                patience -= 1

            print(message)

            if patience == 0:
                print(f"EARLY STOPPING - Model did not learn since {early_stopping} epochs")
                break

        if model_checkpoint and best_weights is not None:
            restore_model_weights(self.layers, best_weights)
            if isinstance(self.optimizer, Adam):
                self.optimizer.t = temp_t
            print(f"MODEL CHECKPOINT: {best_error:.4f}")
            # TODO: Save also optimizer state, better user output

    def evaluate(
        self,
        X: ArrayType,
        y: ArrayType,
        training: bool = False,
        compute_metrics: bool = True,
    ) -> tuple[ArrayType, dict[str, float]]:
        """Evaluates the model on a given batch.

        Args:
            X (ArrayType): Input features.
            y (ArrayType): True labels.
            training (bool): Whether in training mode (affects Dropout/BatchNorm).
            compute_metrics (bool): Whether to calculate extra metrics (Accuracy, etc.).

        Returns:
            tuple[ArrayType, dict[str, float]]: Predictions and dictionary of metric values.
        """
        logits: ArrayType = X.astype(DTYPE, copy=False)
        for layer in self.layers:
            logits = layer.forward(logits, training=training)

        loss: float = self.loss.direct(logits, y)

        metric_dict: dict[str, float] = {}
        metric_dict["loss"] = loss

        predictions_activated: ArrayType = logits
        if self.output_activation is not None:
            predictions_activated = self.output_activation.forward(logits)

        if compute_metrics:
            for metric in self.metrics:
                key = metric.__class__.__name__.lower()

                if isinstance(metric, RMSE) and isinstance(self.loss, MSE):
                    metric_dict[key] = metric.from_mse(loss)
                else:
                    metric_dict[key] = metric(y, predictions_activated)

        predictions: ArrayType = logits
        if not training:
            predictions = predictions_activated

        return predictions, metric_dict

    def test(self, X_test: ArrayType, y_test: ArrayType) -> None:
        """Evaluates the model on the test set and prints results.

        Args:
            X_test (ArrayType): Test features.
            y_test (ArrayType): Test labels.
        """
        X_test = to_device(X_test)
        y_test = to_device(y_test)

        _, metric_dict = self.evaluate(X_test, y_test, training=False)

        print("Test resuls:")
        for key, value in metric_dict.items():
            print(f"   {key} = {value:.4f}")

    def predict(self, X: ArrayType) -> ArrayType:
        """Generates predictions for the input samples.

        Automatically applies the final activation function (Softmax/Sigmoid)
        to return probabilities/values instead of logits.

        Args:
            X (ArrayType): Input features.

        Returns:
            ArrayType: Model predictions (on CPU).
        """
        y: ArrayType = to_device(X.astype(DTYPE, copy=False))
        for layer in self.layers:
            y = layer.forward(y, training=False)

        if self.output_activation is not None:
            y = self.output_activation.forward(y)

        return to_host(y)

    def get_weights(self, optimizer_params: dict | None = None) -> dict:
        """DEPRECATED: Use `mpneuralnetwork.serialization.get_model_weights` instead."""
        import warnings

        warnings.warn(
            "Model.get_weights() is deprecated and will be removed in a future version. "
            "Please use mpneuralnetwork.serialization.get_model_weights(model.layers) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_model_weights(self.layers, optimizer_params)

    def restore_weights(self, weights_dict: dict, optimizer: Optimizer | None = None) -> None:
        """DEPRECATED: Use `mpneuralnetwork.serialization.restore_model_weights` instead."""
        import warnings

        warnings.warn(
            "Model.restore_weights() is deprecated and will be removed in a future version. "
            "Please use mpneuralnetwork.serialization.restore_model_weights(model.layers, ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        restore_model_weights(self.layers, weights_dict, optimizer)
