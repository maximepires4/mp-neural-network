import gzip
import random
from pathlib import Path

import numpy as np
from download_mnist import download_mnist

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import Dense, Dropout
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import Adam


def load_mnist_images(filename):
    if not filename.exists():
        download_mnist()

    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float32) / 255.0


def load_mnist_labels(filename):
    if not filename.exists():
        download_mnist()

    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]


def load_data():
    base_path = Path("data")

    if not base_path.exists() or not (base_path / "train-images-idx3-ubyte.gz").exists():
        print("Data not found. Downloading MNIST...")
        download_mnist()

    X_train = load_mnist_images(base_path / "train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels(base_path / "train-labels-idx1-ubyte.gz")

    X_test = load_mnist_images(base_path / "t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels(base_path / "t10k-labels-idx1-ubyte.gz")

    X_val = X_train[50000:]
    y_val = y_train[50000:]

    X_train = X_train[:50000]
    y_train = y_train[:50000]

    y_train_encoded = one_hot_encode(y_train)
    y_val_encoded = one_hot_encode(y_val)
    y_test_encoded = one_hot_encode(y_test)

    return (X_train, y_train_encoded), (X_val, y_val_encoded), (X_test, y_test_encoded)


if __name__ == "__main__":
    seed = 69
    np.random.seed(seed)
    random.seed(seed)

    print("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()

    print(f"Data loaded. Training on {X_train.shape[0]} samples.")

    network = [Dense(128, input_size=784), ReLU(), Dropout(0.2), Dense(40), ReLU(), Dropout(0.3), Dense(10)]

    model = Model(network, CategoricalCrossEntropy(), Adam())

    model.train(X_train, y_train, epochs=10, batch_size=10, evaluation=(X_val, y_val))

    print("Evaluating on test set...")
    model.test(X_test, y_test)

    Path("output/").mkdir(parents=True, exist_ok=True)
    model.save("output/model.npz")
    print("Model saved to output/model.npz")
