from pathlib import Path

import numpy as np
from dataset import load_mnist

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import (
    BatchNormalization,
    BatchNormalization2D,
    Convolutional,
    Dense,
    Dropout,
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

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist()

    X_train = X_train.reshape(-1, 1, 28, 28)
    X_val = X_val.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    print(f"Train set: {X_train.shape}")

    # Modern architecture: Conv -> BN -> ReLU -> Pool sequence
    network = [
        # Block 1: 28x28 -> 14x14
        Convolutional(output_depth=32, kernel_size=3, input_shape=(1, 28, 28)),
        BatchNormalization2D(),
        ReLU(),
        MaxPooling2D(pool_size=2, strides=2),
        # Block 2: 14x14 -> 7x7
        Convolutional(output_depth=64, kernel_size=3),
        BatchNormalization2D(),
        ReLU(),
        MaxPooling2D(pool_size=2, strides=2),
        # Block 3: Classifier
        Flatten(),
        Dense(128),
        BatchNormalization(),
        ReLU(),
        Dropout(0.5),  # Strong regularization to prevent overfitting
        Dense(10),
    ]

    model = Model(layers=network, loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.001))

    model.train(X_train, y_train, epochs=10, batch_size=64, evaluation=(X_val, y_val), early_stopping=3, model_checkpoint=True)

    print("\nEvaluating on Test Set:")
    model.test(X_test, y_test)

    Path("output/").mkdir(parents=True, exist_ok=True)

    save_model(model, "output/super_cnn_mnist.npz")
    print("Model saved to output/mnist_model.npz")
