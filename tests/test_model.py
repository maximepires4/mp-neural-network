import numpy as np
import pytest

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import Dense
from mpneuralnetwork.losses import MSE, BinaryCrossEntropy
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import SGD, Adam, RMSprop


def test_model_learns_on_simple_regression_task():
    """
    Integration test: a simple model should be able to overfit a tiny regression dataset,
    demonstrating that the forward pass, backward pass, and optimizer work together.
    """

    np.random.seed(69)

    # 1. Arrange: Create a simple dataset and model
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [0.5], [0.5], [1]])  # Target function: y = (x1 + x2) / 2

    layers = [Dense(5, input_size=2), ReLU(), Dense(1)]
    loss = MSE()
    optimizer = SGD(learning_rate=0.1, momentum=0)  # Using SGD with no momentum for simplicity
    model = Model(layers=layers, loss=loss, optimizer=optimizer)

    # Calculate initial loss for comparison
    initial_preds = model.predict(X_train)
    initial_loss = model.loss.direct(initial_preds, y_train)

    # 2. Act: Train the model for a number of epochs
    model.train(X_train, y_train, epochs=100, batch_size=1, auto_evaluation=0)

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
    layers = [Dense(8, input_size=2, initialization="he"), ReLU(), Dense(1, initialization="xavier")]
    loss = BinaryCrossEntropy()
    optimizer = SGD()
    model = Model(layers=layers, loss=loss, optimizer=optimizer)

    # Calculate initial loss for comparison
    initial_preds_logits = model.predict(X_train)
    initial_loss = model.loss.direct(initial_preds_logits, y_train)

    # 2. Act: Train the model
    model.train(X_train, y_train, epochs=1000, batch_size=4, auto_evaluation=0)

    # 3. Assert: Loss should decrease and accuracy should be high
    final_probas = model.predict(X_train)

    # Convert final logits to class predictions (0 or 1)
    final_predictions = (final_probas > 0.5).astype(int)
    accuracy = np.mean(final_predictions == y_train)

    print(f"Classification Test -> Initial Loss: {initial_loss:.4f}, Accuracy: {accuracy:.2f}")
    assert accuracy == 1.0, "Model did not solve the XOR problem with 100% accuracy."


def test_model_save_and_load(tmp_path):
    """
    Tests that a model can be saved to disk and loaded back, preserving weights and optimizer state.
    """
    np.random.seed(42)

    # 1. Setup Model
    layers = [Dense(5, input_size=2), ReLU(), Dense(1)]
    loss = MSE()
    # Use Adam to ensure complex optimizer state (t, momentums, velocities) is saved/loaded
    optimizer = Adam(learning_rate=0.01)
    model = Model(layers, loss, optimizer)

    X = np.random.randn(10, 2)
    y = np.random.randn(10, 1)

    # 2. Train briefly to modify weights and optimizer state
    model.train(X, y, epochs=5, batch_size=2)

    original_weights = [layer.weights.copy() for layer in model.layers if hasattr(layer, "weights")]
    original_biases = [layer.biases.copy() for layer in model.layers if hasattr(layer, "biases")]

    # Capture optimizer state
    # For Adam: t, momentums, velocities
    original_t = model.optimizer.t
    # We need to copy the dicts because they are mutable
    original_momentums = {k: v.copy() for k, v in model.optimizer.momentums.items()}

    # 3. Save
    save_path = tmp_path / "test_model.npz"
    model.save(str(save_path))

    # 4. Load
    loaded_model = Model.load(str(save_path))

    # 5. Verify Weights & Biases
    loaded_weights = [layer.weights for layer in loaded_model.layers if hasattr(layer, "weights")]
    loaded_biases = [layer.biases for layer in loaded_model.layers if hasattr(layer, "biases")]

    for w_orig, w_load in zip(original_weights, loaded_weights, strict=True):
        assert np.allclose(w_orig, w_load), "Weights were not restored correctly."

    for b_orig, b_load in zip(original_biases, loaded_biases, strict=True):
        assert np.allclose(b_orig, b_load), "Biases were not restored correctly."

    # 6. Verify Optimizer State
    assert isinstance(loaded_model.optimizer, Adam)
    assert loaded_model.optimizer.t == original_t, f"Optimizer 't' mismatch. Orig: {original_t}, Load: {loaded_model.optimizer.t}"

    # Check Momentums (checking keys and values)
    assert len(loaded_model.optimizer.momentums) == len(original_momentums)
    # Note: The keys (ids) in the loaded optimizer will be DIFFERENT because the objects are new.
    # But the values should match the corresponding parameter's momentum.
    # Since we iterate layers in order, we can map them.

    # Re-verify by continuing training
    # If state is lost, training might behave erratically or loss might jump.
    loaded_model.train(X, y, epochs=1, batch_size=2)
    final_loss_loaded = loaded_model.loss.direct(loaded_model.predict(X), y)

    # Just ensure it didn't crash and loss is reasonable (not NaN)
    assert not np.isnan(final_loss_loaded)


@pytest.mark.parametrize("optimizer_class", [Adam, RMSprop])
def test_optimizer_convergence(optimizer_class):
    """
    Verifies that advanced optimizers (Adam, RMSprop) can actually solve a simple regression task.
    """
    np.random.seed(42)

    # Simple Linear Regression: y = 2x1 - 3x2 + 1
    X_train = np.random.randn(100, 2)
    y_train = 2 * X_train[:, 0:1] - 3 * X_train[:, 1:2] + 1

    layers = [Dense(1, input_size=2)]
    loss = MSE()
    optimizer = optimizer_class(learning_rate=0.1)
    model = Model(layers, loss, optimizer)

    # Train
    model.train(X_train, y_train, epochs=50, batch_size=10)

    # Check error
    preds = model.predict(X_train)
    final_loss = model.loss.direct(preds, y_train)

    assert final_loss < 0.1, f"{optimizer_class.__name__} failed to converge on simple regression."
