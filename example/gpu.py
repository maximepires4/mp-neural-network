import os
from pathlib import Path

os.environ["MPNN_BACKEND"] = "cupy"
import numpy as np
from dataset import load_mnist

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import (
    BatchNormalization,
    Convolutional,
    Dense,
    Flatten,
    MaxPooling2D,
)
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import Adam
from mpneuralnetwork.serialization import save_model

if __name__ == "__main__":
    print("Classification example with convolution: MNIST Dataset")

    np.random.seed(69)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist(conv=True)

    X_train = X_train[:800]
    y_train = y_train[:800]
    X_val = X_val[:200]
    y_val = y_val[:200]
    X_test = X_test[:100]
    y_test = y_test[:100]

    print(f"Train set: {X_train.shape}")

    network = [
        Convolutional(output_depth=32, kernel_size=3, input_shape=(1, 28, 28)),
        ReLU(),
        MaxPooling2D(),
        Flatten(),
        Dense(128),
        BatchNormalization(),
        ReLU(),
        Dense(10),
    ]

    model = Model(layers=network, loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.001))

    model.train(X_train, y_train, epochs=10, batch_size=64, evaluation=(X_val, y_val), model_checkpoint=False)

    print("\nEvaluating on Test Set:")
    model.test(X_test, y_test)

    Path("output/").mkdir(parents=True, exist_ok=True)

    save_model(model, "output/super_cnn_mnist.npz")
    print("Model saved to output/mnist_model.npz")
