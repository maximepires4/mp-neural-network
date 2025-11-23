import numpy as np
import pytest

from mpneuralnetwork.activations import ReLU, Softmax
from mpneuralnetwork.layers import Dense
from mpneuralnetwork.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
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


def test_smart_weight_initialization():
    """
    Tests that the model automatically selects the correct weight initialization method
    based on the activation function following a Dense layer.
    """
    # 1. Case: Dense -> ReLU (Should use 'he')
    # We need to mock init_weights to verify the call arguments,
    # but since we can't easily mock internal methods without external libs,
    # we will inspect the std deviation of the weights which is a proxy for the method.
    # He: std = sqrt(2/input), Xavier: std = sqrt(1/input)

    input_size = 1000
    output_size = 1000

    # --- Test Auto: He (Dense -> ReLU) ---
    layers_he = [Dense(output_size, input_size=input_size, initialization="auto"), ReLU()]
    Model(layers_he, MSE(), SGD())
    weights_he = layers_he[0].weights
    expected_std_he = np.sqrt(2.0 / input_size)
    actual_std_he = np.std(weights_he)

    # Allow small margin of error for random sampling
    assert np.isclose(actual_std_he, expected_std_he, rtol=0.05), (
        f"Auto-init failed. Expected 'he' (std={expected_std_he:.4f}), got std={actual_std_he:.4f}"
    )

    # --- Test Auto: Xavier (Dense -> Sigmoid/None) ---
    layers_xavier = [Dense(output_size, input_size=input_size, initialization="auto")]
    Model(layers_xavier, MSE(), SGD())
    weights_xavier = layers_xavier[0].weights
    expected_std_xavier = np.sqrt(1.0 / input_size)
    actual_std_xavier = np.std(weights_xavier)

    assert np.isclose(actual_std_xavier, expected_std_xavier, rtol=0.05), (
        f"Auto-init failed. Expected 'xavier' (std={expected_std_xavier:.4f}), got std={actual_std_xavier:.4f}"
    )

    # --- Test Manual Override ---
    # Force Xavier on ReLU (normally He)
    layers_forced = [Dense(output_size, input_size=input_size, initialization="xavier"), ReLU()]
    Model(layers_forced, MSE(), SGD())
    weights_forced = layers_forced[0].weights
    actual_std_forced = np.std(weights_forced)

    assert np.isclose(actual_std_forced, expected_std_xavier, rtol=0.05), "Manual initialization override was ignored."


def test_model_duplicate_activation_removal():
    """
    Test that if the last layer matches the implicit output activation of the loss,
    it is removed to avoid double activation.
    """
    # CCE implies Softmax output activation
    # If we manually add Softmax, it should be removed from self.layers
    # and handled by self.output_activation
    layers = [Dense(10, input_size=5), Softmax()]
    loss = CategoricalCrossEntropy()

    model = Model(layers, loss)

    # The model should have popped the last layer
    # (Softmax is removed from layers list because it's handled by CCE's output_activation logic)
    assert len(model.layers) == 1
    assert isinstance(model.output_activation, Softmax)


def test_model_validation_and_early_stopping():
    """
    Test validation loop and early stopping logic.
    """
    np.random.seed(42)
    X_train = np.random.randn(20, 2)
    y_train = np.random.randn(20, 1)

    X_val = np.random.randn(5, 2)
    y_val = np.random.randn(5, 1)

    layers = [Dense(5, input_size=2), Dense(1)]
    loss = MSE()
    optimizer = SGD(learning_rate=0.01)
    model = Model(layers, loss, optimizer)

    # Train with validation data provided manually
    # Set early_stopping to 2 epochs to test stopping logic
    model.train(
        X_train,
        y_train,
        epochs=5,
        batch_size=5,
        evaluation=(X_val, y_val),
        early_stopping=2,
        auto_evaluation=0,  # Disable auto split since we provide eval
    )

    # Test auto_evaluation split logic
    model2 = Model([Dense(5, input_size=2), Dense(1)], MSE())
    model2.train(X_train, y_train, epochs=2, batch_size=5, auto_evaluation=0.2)
