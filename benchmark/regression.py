import numpy as np

from mpneuralnetwork.activations import ReLU, Sigmoid
from mpneuralnetwork.layers import BatchNormalization, Dense, Dropout
from mpneuralnetwork.losses import MSE
from mpneuralnetwork.metrics import RMSE, R2Score
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import RMSprop

if __name__ == "__main__":
    print("--- Benchmark: Regression (Dense + MSE) ---")
    seed = 42
    np.random.seed(seed)

    N_SAMPLES = 5000
    N_FEATURES = 13

    X_train = np.random.randn(N_SAMPLES, N_FEATURES)
    y_train = (X_train[:, 0] * 2 + X_train[:, 1] * 0.5 + np.random.normal(0, 0.1, N_SAMPLES)).reshape(-1, 1)

    print(f"Training on {X_train.shape[0]} samples.")

    network = [
        Dense(64, input_size=N_FEATURES),
        BatchNormalization(),
        ReLU(),
        Dense(128),
        ReLU(),
        Dropout(0.2),
        Dense(64),
        ReLU(),
        Dense(32),
        Sigmoid(),
        Dense(1),
    ]

    model = Model(network, MSE(), RMSprop(learning_rate=0.001), metrics=[RMSE(), R2Score()])

    model.train(X_train, y_train, epochs=5, batch_size=64, auto_evaluation=0.0, model_checkpoint=False)
