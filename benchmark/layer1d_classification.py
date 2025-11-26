import numpy as np

from example.dataset import load_mnist
from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import Dense, Dropout
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import Adam

if __name__ == "__main__":
    print("--- Benchmark: 1D MLP (Dense + Dropout) ---")
    seed = 69
    np.random.seed(seed)

    print("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist()

    train_indices = np.random.permutation(1000)
    val_indices = np.random.permutation(200)
    test_indices = np.random.permutation(100)

    X_train_s, y_train_s = X_train[train_indices], y_train[train_indices]
    X_val_s, y_val_s = X_val[val_indices], y_val[val_indices]
    X_test_s, y_test_s = X_test[test_indices], y_test[test_indices]

    print(f"Data loaded. Training on {X_train.shape[0]} samples.")

    network = [Dense(128, input_size=784), ReLU(), Dropout(0.2), Dense(40), ReLU(), Dropout(0.3), Dense(10)]

    model = Model(network, CategoricalCrossEntropy(), Adam())

    model.train(X_train_s, y_train_s, epochs=3, batch_size=32, auto_evaluation=0.0, model_checkpoint=False)
