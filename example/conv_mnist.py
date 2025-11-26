from pathlib import Path

import numpy as np
from dataset import load_mnist

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import BatchNormalization, Convolutional, Dense, Flatten, MaxPooling2D
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import Adam
from mpneuralnetwork.serialization import save_model

if __name__ == "__main__":
    print("Classification example with convolution: MNIST Dataset")
    seed = 69
    np.random.seed(seed)

    print("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist(conv=True)

    print(f"Data loaded. Training on {X_train.shape[0]} samples.")
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

    model = Model(network, CategoricalCrossEntropy(), Adam())

    model.train(X_train, y_train, epochs=10, batch_size=64, evaluation=(X_val, y_val))

    print("Evaluating on test set...")
    model.test(X_test, y_test)

    Path("output/").mkdir(parents=True, exist_ok=True)

    save_model(model, "output/mnist_model.npz")
    print("Model saved to output/mnist_model.npz")
