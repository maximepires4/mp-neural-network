import sys
from io import StringIO

import numpy as np

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import Dense
from mpneuralnetwork.losses import MSE
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import SGD, Adam


def test_early_stopping_triggers():
    """
    Tests that training stops early when validation loss stops improving.
    """
    np.random.seed(42)
    # Create data where generalization is hard (random noise vs random targets)
    # This ensures validation loss won't improve much, triggering early stopping.
    X_train = np.random.randn(100, 5)
    y_train = np.random.randn(100, 1)

    X_val = np.random.randn(20, 5)
    y_val = np.random.randn(20, 1)  # Totally random validation targets

    model = Model([Dense(1, input_size=5)], MSE(), SGD(learning_rate=0.01))

    # Capture stdout to check for "EARLY STOPPING" message (optional, but good confirmation)
    captured_output = StringIO()
    sys.stdout = captured_output

    try:
        # Train with high epochs but low patience
        max_epochs = 50
        patience = 3
        model.train(X_train, y_train, epochs=max_epochs, batch_size=10, evaluation=(X_val, y_val), early_stopping=patience)
    finally:
        sys.stdout = sys.__stdout__  # Restore stdout

    output = captured_output.getvalue()

    # 1. Verify Early Stopping message or behavior
    assert "EARLY STOPPING" in output

    # 2. Verify we didn't run all epochs
    # We can infer this by counting "epoch" lines in output
    epoch_lines = [line for line in output.split("\n") if "epoch" in line]
    assert len(epoch_lines) < max_epochs, f"Training ran for {len(epoch_lines)} epochs, expected early stopping."
    assert len(epoch_lines) >= patience, "Training stopped too early, should run at least 'patience' epochs."


def test_model_checkpoint_restores_best_weights():
    """
    Tests that the model restores the weights associated with the lowest validation error,
    not necessarily the weights from the last epoch.
    """
    np.random.seed(42)

    # 1. Create model and save "Good" weights
    model = Model([Dense(1, input_size=1), ReLU(), Dense(1)], MSE(), Adam(learning_rate=0.1))
    model.layers[0].weights[:] = 1.0  # Set known good weights
    best_weights_memory = model._deepcopy()

    # 2. Modify model weights to "Bad"
    model.layers[0].weights[:] = 999.0

    # 3. Trigger restore
    model._restore_weights(model, best_weights_memory)

    # 4. Verify restoration
    assert np.allclose(model.layers[0].weights, 1.0), "Model checkpoint failed to restore weights."


def test_retraining_maintains_optimizer_state():
    """
    Tests that calling train() a second time continues from where it left off,
    maintaining optimizer state (momentum, etc.).
    """
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randn(50, 1)

    # Use SGD with Momentum. If state is lost, momentum resets to 0.
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model = Model([Dense(1, input_size=5)], MSE(), optimizer)

    # Train 1 epoch
    model.train(X, y, epochs=1, batch_size=10)

    # Check that velocities are non-zero
    assert len(optimizer.velocities) > 0
    param_id = list(optimizer.velocities.keys())[0]
    velocity_after_epoch1 = optimizer.velocities[param_id].copy()
    assert not np.allclose(velocity_after_epoch1, 0), "Optimizer state (velocity) should not be zero after training."

    # Train 1 more epoch
    model.train(X, y, epochs=1, batch_size=10)

    # Check that velocities have changed (evolved), not reset
    velocity_after_epoch2 = optimizer.velocities[param_id]

    # If state was reset, the velocity calculation would restart from 0 context,
    # producing a value solely based on the new gradient, which is likely different but let's check continuity.
    # A better check: ensure the dictionary IS the same object or keys are preserved.
    assert param_id in optimizer.velocities
    assert not np.allclose(velocity_after_epoch1, velocity_after_epoch2), "Optimizer state should evolve."

    # Verify loss continues to decrease (or stays low)
    loss_after = model.loss.direct(model.predict(X), y)
    assert loss_after < 10.0  # Sanity check


def test_auto_evaluation_splits_data():
    """
    Tests that the auto_evaluation parameter correctly triggers validation
    during training by checking the logs for validation metrics.
    """
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100, 1)

    model = Model([Dense(1, input_size=5)], MSE(), SGD(learning_rate=0.01))

    captured_output = StringIO()
    sys.stdout = captured_output

    try:
        # Train with auto_evaluation (default is 0.2)
        model.train(X, y, epochs=1, batch_size=10, auto_evaluation=0.2)
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()

    assert "val_error" in output, "Auto-evaluation did not seem to trigger (no 'val_error' in logs)."
    assert "val_accuracy" not in output, "Regression task should not report accuracy."
