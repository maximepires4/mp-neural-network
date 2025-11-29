import numpy as np
import pytest

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import Convolutional, Dense, Flatten
from mpneuralnetwork.losses import MSE
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import Adam
from mpneuralnetwork.serialization import get_model_weights, load_model, save_model


def test_model_save_and_load(tmp_path):
    np.random.seed(42)
    layers = [Dense(5, input_size=2), ReLU(), Dense(1)]
    loss = MSE()
    optimizer = Adam(learning_rate=0.01)
    model = Model(layers, loss, optimizer)

    X = np.random.randn(10, 2).astype(np.float32)
    y = np.random.randn(10, 1).astype(np.float32)

    model.train(X, y, epochs=5, batch_size=2)

    # Capture state using the correct API (returns host numpy arrays)
    original_state = get_model_weights(model.layers, model.optimizer.params)
    original_t = model.optimizer.t

    save_path = tmp_path / "test_model.npz"
    save_model(model, str(save_path))

    loaded_model = load_model(str(save_path))

    # Capture loaded state
    loaded_state = get_model_weights(loaded_model.layers, loaded_model.optimizer.params)

    # 1. Verify Weights and Optimizer State
    for key in original_state:
        assert key in loaded_state
        assert np.allclose(original_state[key], loaded_state[key]), f"Mismatch in {key}"

    # 2. Verify Optimizer Attributes
    assert isinstance(loaded_model.optimizer, Adam)
    assert loaded_model.optimizer.t == original_t

    # 3. Verify Functional Continuity
    # Run a training step to ensure everything is connected (grads, optimizer references)
    try:
        loaded_model.train(X, y, epochs=1, batch_size=2)
    except Exception as e:
        pytest.fail(f"Training failed after loading: {e}")

    final_loss_loaded = loaded_model.loss.direct(loaded_model.predict(X), y)
    assert not np.isnan(final_loss_loaded)


def test_conv_model_save_load_resume(tmp_path):
    """
    Test d'intégration: Conv -> Save -> Load -> Resume Training.
    """
    input_shape = (1, 10, 10)
    X = np.random.randn(10, *input_shape).astype(np.float32)
    y = np.random.randn(10, 1).astype(np.float32)  # Regression target

    # Conv (output 2x8x8) -> Flatten (128) -> Dense (1)
    network = [Convolutional(output_depth=2, kernel_size=3, input_shape=input_shape, initialization="he"), Flatten(), Dense(1)]

    model = Model(network, MSE(), Adam(learning_rate=0.01))

    # 1. Train briefly
    model.train(X, y, epochs=2, batch_size=2)

    original_weights = get_model_weights(model.layers)

    # 2. Save
    save_path = tmp_path / "conv_test_model.npz"
    save_model(model, str(save_path))

    # 3. Load
    loaded_model = load_model(str(save_path))
    loaded_weights = get_model_weights(loaded_model.layers)

    # 4. Verify Weights
    for key in original_weights:
        assert np.allclose(original_weights[key], loaded_weights[key]), f"Mismatch in {key}"

    # 5. Resume Training
    # We pick a specific weight to watch
    # Note: get_model_weights returns copies, so we can use it to compare 'before' vs 'after'
    initial_loaded_weights = loaded_weights["layer_2_weights"]

    loaded_model.train(X, y, epochs=1, batch_size=2)

    new_loaded_weights = get_model_weights(loaded_model.layers)["layer_2_weights"]

    # Weights should have moved
    assert not np.allclose(initial_loaded_weights, new_loaded_weights)
