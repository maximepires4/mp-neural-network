import sys
from io import StringIO

import numpy as np

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import Dense
from mpneuralnetwork.losses import MSE
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import SGD, Adam
from mpneuralnetwork.serialization import get_model_weights, restore_model_weights


def test_early_stopping_triggers():
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randn(100, 1)
    X_val = np.random.randn(20, 5)
    y_val = np.random.randn(20, 1)

    model = Model([Dense(1, input_size=5)], MSE(), SGD(learning_rate=0.01))

    captured_output = StringIO()
    sys.stdout = captured_output

    max_epochs = 50
    patience = 3
    try:
        model.train(X_train, y_train, epochs=max_epochs, batch_size=10, evaluation=(X_val, y_val), early_stopping=patience)
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    assert "EARLY STOPPING" in output
    epoch_lines = [line for line in output.split("\n") if "epoch" in line]
    assert len(epoch_lines) < max_epochs
    assert len(epoch_lines) >= patience


def test_model_checkpoint_restores_best_weights():
    np.random.seed(42)
    model = Model([Dense(1, input_size=1), ReLU(), Dense(1)], MSE(), Adam(learning_rate=0.1))
    model.layers[0].weights[:] = 1.0
    best_weights_memory = get_model_weights(model.layers)

    model.layers[0].weights[:] = -100.0
    restore_model_weights(model.layers, best_weights_memory)

    assert np.allclose(model.layers[0].weights, 1.0)


def test_checkpoint_restores_optimizer_state():
    """
    Tests that restore_weights correctly restores the optimizer state (e.g., Adam's 't'),
    ensuring that the mechanism used by ModelCheckpoint works technically.
    """
    return  # TODO: temp_t in model
    np.random.seed(69)
    optimizer = Adam(learning_rate=0.1)
    model = Model([Dense(1, input_size=2)], MSE(), optimizer)

    # 1. Simulate a training state
    model.optimizer.t = 10
    model.optimizer.momentums = {id(layer.weights): np.ones_like(layer.weights) for layer in model.layers}

    # 2. Save this "Best" state
    # We explicitly use the same method as Model.train to capture the state
    best_weights = model.get_weights(model.optimizer.params)

    # 3. Ruin the state (Simulate further training drifting away)
    model.optimizer.t = 999
    for m in model.optimizer.momentums.values():
        m[:] = -5.0

    # 4. Trigger restore
    model.restore_weights(best_weights, model.optimizer)

    # 5. Verify restoration
    assert model.optimizer.t == 10, f"Optimizer step 't' was not restored. Expected 10, got {model.optimizer.t}"

    # Check a momentum value (taking the first one found)
    first_momentum = next(iter(model.optimizer.momentums.values()))
    assert np.allclose(first_momentum, 1.0), "Optimizer momentums were not restored."
