import numpy as np
import pytest

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import Dense
from mpneuralnetwork.losses import MSE, BinaryCrossEntropy
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import SGD, Adam, RMSprop


def test_model_learns_on_simple_regression_task():
    np.random.seed(69)
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [0.5], [0.5], [1]])

    layers = [Dense(5, input_size=2), ReLU(), Dense(1)]
    loss = MSE()
    optimizer = SGD(learning_rate=0.1, momentum=0)
    model = Model(layers=layers, loss=loss, optimizer=optimizer)

    initial_loss = model.loss.direct(model.predict(X_train), y_train)
    model.train(X_train, y_train, epochs=100, batch_size=1, auto_evaluation=0)
    final_loss = model.loss.direct(model.predict(X_train), y_train)

    assert final_loss < initial_loss / 5


def test_model_learns_on_binary_classification_task():
    np.random.seed(69)
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    layers = [Dense(8, input_size=2, initialization="he"), ReLU(), Dense(1, initialization="xavier")]
    loss = BinaryCrossEntropy()
    optimizer = SGD()
    model = Model(layers=layers, loss=loss, optimizer=optimizer)

    model.train(X_train, y_train, epochs=1000, batch_size=4, auto_evaluation=0)

    final_probas = model.predict(X_train)
    final_predictions = (final_probas > 0.5).astype(int)
    accuracy = np.mean(final_predictions == y_train)
    assert accuracy == 1.0


def test_model_validation_and_early_stopping():
    np.random.seed(42)
    X_train = np.random.randn(20, 2)
    y_train = np.random.randn(20, 1)
    X_val = np.random.randn(5, 2)
    y_val = np.random.randn(5, 1)

    layers = [Dense(5, input_size=2), Dense(1)]
    model = Model(layers, MSE(), SGD(learning_rate=0.01))

    model.train(X_train, y_train, epochs=5, batch_size=5, evaluation=(X_val, y_val), early_stopping=2, auto_evaluation=0)

    model2 = Model([Dense(5, input_size=2), Dense(1)], MSE())
    model2.train(X_train, y_train, epochs=2, batch_size=5, auto_evaluation=0.2)


@pytest.mark.parametrize("optimizer_class", [Adam, RMSprop])
def test_optimizer_convergence(optimizer_class):
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = 2 * X_train[:, 0:1] - 3 * X_train[:, 1:2] + 1

    layers = [Dense(1, input_size=2)]
    loss = MSE()
    optimizer = optimizer_class(learning_rate=0.1)
    model = Model(layers, loss, optimizer)

    model.train(X_train, y_train, epochs=50, batch_size=10)
    final_loss = model.loss.direct(model.predict(X_train), y_train)
    assert final_loss < 0.1
