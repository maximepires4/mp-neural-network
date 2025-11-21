import numpy as np

from mpneuralnetwork.layers import Dense
from mpneuralnetwork.losses import MSE
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import SGD


def generate_data(n_samples=1000):
    """
    Génère des données synthétiques pour une régression linéaire simple : y = 3x + 2 + bruit
    """
    X = np.random.rand(n_samples, 1) * 10  # X entre 0 et 10
    noise = np.random.randn(n_samples, 1) * 0.5
    y = 3 * X + 2 + noise
    return X, y


if __name__ == "__main__":
    seed = 69
    np.random.seed(seed)

    print("Génération des données synthétiques (y = 3x + 2)...")
    X, y = generate_data(1200)

    X_train, y_train = X[:800], y[:800]
    X_val, y_val = X[800:1000], y[800:1000]
    X_test, y_test = X[1000:], y[1000:]

    network = [Dense(1, input_size=1)]

    model = Model(network, MSE(), SGD(learning_rate=0.001, momentum=0.9))

    model.train(X_train, y_train, epochs=50, batch_size=32, evaluation=(X_val, y_val))

    model.test(X_test, y_test)

    learned_slope = model.layers[0].weights[0][0]
    learned_intercept = model.layers[0].biases[0][0]

    print("Modèle mathématique     : y = 3.00 * x + 2.00")
    print(f"Modèle appris           : y = {learned_slope:.2f} * x + {learned_intercept:.2f}")
