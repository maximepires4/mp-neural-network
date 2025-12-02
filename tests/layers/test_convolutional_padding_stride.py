import numpy as np

from mpneuralnetwork.layers import Convolutional


def test_conv_padding_valid():
    """Test 'valid' padding (no padding)."""
    batch_size = 2
    input_depth = 1
    h, w = 10, 10
    k = 3

    layer = Convolutional(output_depth=1, kernel_size=k, input_shape=(input_depth, h, w), padding="valid", initialization="xavier")

    x = np.random.randn(batch_size, input_depth, h, w)
    out = layer.forward(x)

    # Expected: (10 - 3 + 0) / 1 + 1 = 8
    assert out.shape == (batch_size, 1, 8, 8)
    assert layer.padding == 0


def test_conv_padding_same_stride_1():
    """Test 'same' padding with stride 1. Output size should equal input size."""
    batch_size = 2
    input_depth = 1
    h, w = 10, 10
    k = 3  # Padding should be (3-1)//2 = 1

    layer = Convolutional(output_depth=1, kernel_size=k, input_shape=(input_depth, h, w), padding="same", initialization="xavier")

    x = np.random.randn(batch_size, input_depth, h, w)
    out = layer.forward(x)

    assert out.shape == (batch_size, 1, 10, 10)
    assert layer.padding == 1


def test_conv_padding_same_stride_2():
    """Test 'same' padding with stride 2. Output size should be halved."""
    batch_size = 2
    input_depth = 1
    h, w = 10, 10
    k = 3
    stride = 2

    layer = Convolutional(output_depth=1, kernel_size=k, input_shape=(input_depth, h, w), padding="same", stride=stride, initialization="xavier")

    x = np.random.randn(batch_size, input_depth, h, w)
    out = layer.forward(x)

    # P=1. (10 - 3 + 2*1) // 2 + 1 = 9 // 2 + 1 = 4 + 1 = 5
    assert out.shape == (batch_size, 1, 5, 5)


def test_conv_padding_integer():
    """Test manual integer padding."""
    batch_size = 2
    input_depth = 1
    h, w = 10, 10
    k = 3
    pad = 2

    layer = Convolutional(output_depth=1, kernel_size=k, input_shape=(input_depth, h, w), padding=pad, initialization="xavier")

    x = np.random.randn(batch_size, input_depth, h, w)
    out = layer.forward(x)

    # (10 - 3 + 2*2) // 1 + 1 = 11 // 1 + 1 = 12
    assert out.shape == (batch_size, 1, 12, 12)
    assert layer.padding == 2


def test_conv_stride_only():
    """Test stride > 1 with valid padding."""
    batch_size = 2
    input_depth = 1
    h, w = 10, 10
    k = 3
    stride = 2

    layer = Convolutional(output_depth=1, kernel_size=k, input_shape=(input_depth, h, w), padding="valid", stride=stride, initialization="xavier")

    x = np.random.randn(batch_size, input_depth, h, w)
    out = layer.forward(x)

    # (10 - 3 + 0) // 2 + 1 = 7 // 2 + 1 = 3 + 1 = 4
    assert out.shape == (batch_size, 1, 4, 4)


def test_conv_backward_shape_with_padding():
    """Test that backward pass returns correct shape gradient with padding."""
    batch_size = 2
    input_depth = 1
    h, w = 10, 10
    k = 3

    layer = Convolutional(output_depth=2, kernel_size=k, input_shape=(input_depth, h, w), padding="same", initialization="xavier")

    x = np.random.randn(batch_size, input_depth, h, w)
    out = layer.forward(x)

    grad_out = np.random.randn(*out.shape)
    grad_in = layer.backward(grad_out)

    # Gradient should have same shape as input (padding removed)
    assert grad_in.shape == x.shape


def test_conv_backward_shape_with_stride():
    """Test that backward pass returns correct shape gradient with stride."""
    batch_size = 2
    input_depth = 1
    h, w = 10, 10
    k = 3
    stride = 2

    layer = Convolutional(output_depth=2, kernel_size=k, input_shape=(input_depth, h, w), padding="same", stride=stride, initialization="xavier")

    x = np.random.randn(batch_size, input_depth, h, w)
    out = layer.forward(x)

    grad_out = np.random.randn(*out.shape)
    grad_in = layer.backward(grad_out)

    assert grad_in.shape == x.shape
