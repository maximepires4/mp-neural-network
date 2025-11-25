import numpy as np

from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import Convolutional, Dense, Flatten
from mpneuralnetwork.losses import MSE
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import Adam
from mpneuralnetwork.serialization import load_model, save_model


def test_model_save_and_load(tmp_path):
    np.random.seed(42)
    layers = [Dense(5, input_size=2), ReLU(), Dense(1)]
    loss = MSE()
    optimizer = Adam(learning_rate=0.01)
    model = Model(layers, loss, optimizer)

    X = np.random.randn(10, 2)
    y = np.random.randn(10, 1)

    model.train(X, y, epochs=5, batch_size=2)

    original_weights = [layer.weights.copy() for layer in model.layers if hasattr(layer, "weights")]
    original_biases = [layer.biases.copy() for layer in model.layers if hasattr(layer, "biases")]
    original_t = model.optimizer.t
    original_momentums = {k: v.copy() for k, v in model.optimizer.momentums.items()}

    save_path = tmp_path / "test_model.npz"
    save_model(model, str(save_path))

    loaded_model = load_model(str(save_path))

    loaded_weights = [layer.weights for layer in loaded_model.layers if hasattr(layer, "weights")]
    loaded_biases = [layer.biases for layer in loaded_model.layers if hasattr(layer, "biases")]

    for w_orig, w_load in zip(original_weights, loaded_weights, strict=True):
        assert np.allclose(w_orig, w_load)

    for b_orig, b_load in zip(original_biases, loaded_biases, strict=True):
        assert np.allclose(b_orig, b_load)

    assert isinstance(loaded_model.optimizer, Adam)
    assert loaded_model.optimizer.t == original_t
    assert len(loaded_model.optimizer.momentums) == len(original_momentums)

    loaded_model.train(X, y, epochs=1, batch_size=2)
    final_loss_loaded = loaded_model.loss.direct(loaded_model.predict(X), y)
    assert not np.isnan(final_loss_loaded)


def test_conv_model_save_load_resume(tmp_path):
    """
    Test d'intégration: Conv -> Save -> Load -> Resume Training.
    """
    input_shape = (1, 10, 10)
    X = np.random.randn(10, *input_shape)
    y = np.random.randn(10, 1)  # Regression target

    # Conv (output 8x8) -> Flatten (64) -> Dense (1)
    network = [Convolutional(output_depth=2, kernel_size=3, input_shape=input_shape, initialization="he"), Flatten(), Dense(1)]

    model = Model(network, MSE(), Adam(learning_rate=0.01))

    # 2. Train briefly
    model.train(X, y, epochs=2, batch_size=2)

    original_weights = model.get_weights()

    # 3. Save
    save_path = tmp_path / "conv_test_model.npz"
    save_model(model, str(save_path))

    # 4. Load
    loaded_model = load_model(str(save_path))
    loaded_weights = loaded_model.get_weights()

    # 5. Verify Weights
    for key in original_weights:
        assert np.allclose(original_weights[key], loaded_weights[key]), f"Mismatch in {key}"

    # 6. Resume Training
    initial_loaded_weights = loaded_model.get_weights()["layer_2_weights"].copy()  # Dense weights

    loaded_model.train(X, y, epochs=1, batch_size=2)

    new_loaded_weights = loaded_model.get_weights()["layer_2_weights"]

    # Weights should have moved
    assert not np.allclose(initial_loaded_weights, new_loaded_weights)
