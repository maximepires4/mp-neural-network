from pathlib import Path

import numpy as np
from dataset import load_california

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import BatchNormalization, Dense, Dropout
from mpneuralnetwork.losses import MSE
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import Adam

if __name__ == "__main__":
    print("Regression example: California Housing Dataset")
    seed = 69
    np.random.seed(seed)

    print("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_california()

    print("Data loaded.")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features per sample: {X_train.shape[1]}")

    network = [
        Dense(256, input_size=8),
        BatchNormalization(),
        ReLU(),
        Dropout(0.2),
        Dense(128),
        BatchNormalization(),
        ReLU(),
        Dropout(0.1),
        Dense(64),
        BatchNormalization(),
        ReLU(),
        Dense(32),
        ReLU(),
        Dense(1),
    ]

    optimizer = Adam(learning_rate=0.001)
    model = Model(network, MSE(), optimizer)

    model.train(
        X_train,
        y_train,
        epochs=100,
        batch_size=64,
        evaluation=(X_val, y_val),
        early_stopping=10,
        model_checkpoint=True,
    )

    print("Evaluating on test set...")
    model.test(X_test, y_test)

    Path("output/").mkdir(parents=True, exist_ok=True)
    model.save("output/california_model.npz")
    print("Model saved to output/california_model.npz")
