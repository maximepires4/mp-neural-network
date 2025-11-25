import numpy as np
import pytest

from mpneuralnetwork.layers import BatchNormalization, Dense, Dropout, Layer


def test_layer_base_methods():
    """Test get_config and build for layers."""
    layer = Layer()
    assert layer.get_config() == {"type": "Layer"}
    layer.build(10)
    assert layer.input_size == 10
    assert layer.output_size == 10
    assert layer.params == {}


@pytest.mark.parametrize(
    "layer, input_shape, expected_output_shape",
    [
        (Dense(5, input_size=10, initialization="xavier"), (64, 10), (64, 5)),
        (Dense(1, input_size=3, initialization="xavier"), (1, 3), (1, 1)),
        (Dropout(0.5), (128, 20), (128, 20)),
        (BatchNormalization(), (32, 10), (32, 10)),
    ],
)
def test_layer_output_shapes(layer, input_shape, expected_output_shape):
    """
    Tests that the output shape of a layer's forward pass is correct.
    """
    if isinstance(layer, BatchNormalization):
        layer.build(input_shape[1])

    input_data = np.random.randn(*input_shape)
    output = layer.forward(input_data)

    assert output.shape == expected_output_shape, (
        f"Shape mismatch for layer {layer.__class__.__name__}. Input: {input_shape}, Output: {output.shape}, Expected: {expected_output_shape}"
    )
