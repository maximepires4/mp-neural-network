import random
from pathlib import Path

import numpy as np
from dataset import load_mnist

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import Dense, Dropout
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import Adam

if __name__ == "__main__":
    print("Classification example: MNIST Dataset")
    seed = 69
    np.random.seed(seed)
    random.seed(seed)

    print("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist()

    print(f"Data loaded. Training on {X_train.shape[0]} samples.")

    network = [Dense(128, input_size=784), ReLU(), Dropout(0.2), Dense(40), ReLU(), Dropout(0.3), Dense(10)]

    model = Model(network, CategoricalCrossEntropy(), Adam())

    model.train(X_train, y_train, epochs=10, batch_size=10, evaluation=(X_val, y_val))

    print("Evaluating on test set...")
    model.test(X_test, y_test)

    Path("output/").mkdir(parents=True, exist_ok=True)

    model.save("output/mnist_model.npz")
    print("Model saved to output/mnist_model.npz")
