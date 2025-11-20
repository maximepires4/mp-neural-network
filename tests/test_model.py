import numpy as np
import pytest
from mpneuralnetwork.model import Model
from mpneuralnetwork.layers import Dense
from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.losses import MSE, BinaryCrossEntropy
from mpneuralnetwork.optimizers import SGD


def test_model_learns_on_simple_regression_task():
    """
    Integration test: a simple model should be able to overfit a tiny regression dataset,
    demonstrating that the forward pass, backward pass, and optimizer work together.
    """
    
    np.random.seed(69)

    # 1. Arrange: Create a simple dataset and model
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [0.5], [0.5], [1]])  # Target function: y = (x1 + x2) / 2

    layers = [Dense(2, 5), ReLU(), Dense(5, 1)]
    loss = MSE()
    optimizer = SGD(learning_rate=0.1, momentum=0) # Using SGD with no momentum for simplicity
    model = Model(layers=layers, loss=loss, optimizer=optimizer)

    # Calculate initial loss for comparison
    initial_preds = model.predict(X_train)
    initial_loss = model.loss.direct(initial_preds, y_train)

    # 2. Act: Train the model for a number of epochs
    model.train(X_train, y_train, epochs=100, batch_size=1)

    # 3. Assert: The final loss should be significantly lower than the
    final_preds = model.predict(X_train)
    final_loss = model.loss.direct(final_preds, y_train)

    print(f"Regression Test -> Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss / 5, "Model did not learn; loss did not decrease significantly."


def test_model_learns_on_binary_classification_task():
    """
    Integration test: a simple model should be able to solve the XOR problem,
    a classic non-linear binary classification task.
    """

    np.random.seed(69)

    # 1. Arrange: Create the XOR dataset and a suitable model
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    # The model does not need a final Sigmoid layer, as BinaryCrossEntropy loss expects raw logits.
    layers = [Dense(2, 8, 'he'), ReLU(), Dense(8, 1, 'xavier')]
    loss = BinaryCrossEntropy()
    optimizer = SGD()
    model = Model(layers=layers, loss=loss, optimizer=optimizer)

    # Calculate initial loss for comparison
    initial_preds_logits = model.predict(X_train)
    initial_loss = model.loss.direct(initial_preds_logits, y_train)

    # 2. Act: Train the model
    model.train(X_train, y_train, epochs=1000, batch_size=4)

    # 3. Assert: Loss should decrease and accuracy should be high
    final_probas = model.predict(X_train)

    # Convert final logits to class predictions (0 or 1)
    final_predictions = (final_probas > 0.5).astype(int)
    accuracy = np.mean(final_predictions == y_train)

    print(f"Classification Test -> Initial Loss: {initial_loss:.4f}, Accuracy: {accuracy:.2f}")
    assert accuracy == 1.0, "Model did not solve the XOR problem with 100% accuracy."
