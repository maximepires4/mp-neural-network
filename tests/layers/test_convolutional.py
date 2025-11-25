import numpy as np

from mpneuralnetwork.layers import Convolutional


def test_conv_output_shape():
    batch_size = 10
    input_depth = 3
    input_h, input_w = 32, 32
    output_depth = 16
    kernel_size = 3

    layer = Convolutional(output_depth, kernel_size, input_shape=(input_depth, input_h, input_w), initialization="xavier")

    X = np.random.randn(batch_size, input_depth, input_h, input_w)
    output = layer.forward(X)

    expected_h = input_h - kernel_size + 1
    expected_w = input_w - kernel_size + 1

    assert output.shape == (batch_size, output_depth, expected_h, expected_w)


def test_conv_forward_values():
    X = np.ones((1, 1, 3, 3))
    layer = Convolutional(output_depth=1, kernel_size=2, input_shape=(1, 3, 3), initialization="xavier", no_bias=True)

    layer.kernels = np.ones_like(layer.kernels)

    output = layer.forward(X)

    assert output.shape == (1, 1, 2, 2)
    assert np.allclose(output, 4.0)


def test_conv_backward_gradient_shape():
    batch_size = 5
    input_depth = 2
    H, W = 10, 10
    out_depth = 4
    k = 3

    layer = Convolutional(out_depth, k, input_shape=(input_depth, H, W), initialization="xavier")

    X = np.random.randn(batch_size, input_depth, H, W)

    out = layer.forward(X)
    dout = np.random.randn(*out.shape)
    dx = layer.backward(dout)

    assert dx.shape == X.shape
    assert layer.kernels_gradient.shape == layer.kernels.shape
    assert layer.biases_gradient.shape == layer.biases.shape

    assert np.abs(layer.kernels_gradient).sum() > 0
    assert np.abs(layer.biases_gradient).sum() > 0
    assert np.abs(dx).sum() > 0


def test_conv_gradient_value_check():
    X = np.random.randn(1, 1, 5, 5)
    layer = Convolutional(output_depth=1, kernel_size=3, input_shape=(1, 5, 5), no_bias=True, initialization="xavier")

    out = layer.forward(X)
    dout = np.ones_like(out)
    dx_custom = layer.backward(dout)

    kernel = layer.kernels[0, 0]
    assert np.isclose(dx_custom[0, 0, 2, 2], np.sum(kernel))
