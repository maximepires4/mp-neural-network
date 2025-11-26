import numpy as np

from mpneuralnetwork.layers.layer2d import AveragePooling2D, MaxPooling2D


class TestPooling:
    def test_max_pooling_output_shape(self):
        batch_size = 10
        channels = 3
        h, w = 32, 32
        pool_size = 2
        stride = 2

        layer = MaxPooling2D(pool_size=pool_size, strides=stride)
        layer.build((channels, h, w))

        X = np.random.randn(batch_size, channels, h, w)
        out = layer.forward(X)

        expected_h = (h - pool_size) // stride + 1
        expected_w = (w - pool_size) // stride + 1

        assert out.shape == (batch_size, channels, expected_h, expected_w)

    def test_max_pooling_forward_values(self):
        # Manual check: 2x2 input, pool 2, stride 2
        # Input: [[1, 2], [3, 4]] -> Max: 4
        X = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        layer = MaxPooling2D(pool_size=2, strides=2)
        layer.build((1, 2, 2))

        out = layer.forward(X)

        assert out.shape == (1, 1, 1, 1)
        assert out[0, 0, 0, 0] == 4.0

    def test_max_pooling_backward_shape(self):
        batch_size = 5
        channels = 2
        h, w = 10, 10
        layer = MaxPooling2D(pool_size=2, strides=2)
        layer.build((channels, h, w))

        X = np.random.randn(batch_size, channels, h, w)
        out = layer.forward(X)
        dout = np.random.randn(*out.shape)
        dx = layer.backward(dout)

        assert dx.shape == X.shape

    def test_max_pooling_backward_values(self):
        # Gradient should only pass through the max value
        X = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        layer = MaxPooling2D(pool_size=2, strides=2)
        layer.build((1, 2, 2))

        out = layer.forward(X)
        dout = np.ones_like(out)  # grad 1.0
        dx = layer.backward(dout)

        # Only index (1,1) (value 4.0) should receive gradient
        expected_dx = np.array([[[[0.0, 0.0], [0.0, 1.0]]]])

        assert np.allclose(dx, expected_dx)

    def test_avg_pooling_output_shape(self):
        batch_size = 10
        channels = 3
        h, w = 32, 32
        pool_size = 2
        stride = 2

        layer = AveragePooling2D(pool_size=pool_size, strides=stride)
        layer.build((channels, h, w))

        X = np.random.randn(batch_size, channels, h, w)
        out = layer.forward(X)

        expected_h = (h - pool_size) // stride + 1
        expected_w = (w - pool_size) // stride + 1

        assert out.shape == (batch_size, channels, expected_h, expected_w)

    def test_avg_pooling_forward_values(self):
        # Manual check: 2x2 input, pool 2, stride 2
        # Input: [[1, 3], [3, 5]] -> Mean: (1+3+3+5)/4 = 12/4 = 3
        X = np.array([[[[1.0, 3.0], [3.0, 5.0]]]])
        layer = AveragePooling2D(pool_size=2, strides=2)
        layer.build((1, 2, 2))

        out = layer.forward(X)

        assert out.shape == (1, 1, 1, 1)
        assert np.isclose(out[0, 0, 0, 0], 3.0)

    def test_avg_pooling_backward_shape(self):
        batch_size = 5
        channels = 2
        h, w = 10, 10
        layer = AveragePooling2D(pool_size=2, strides=2)
        layer.build((channels, h, w))

        X = np.random.randn(batch_size, channels, h, w)
        out = layer.forward(X)
        dout = np.random.randn(*out.shape)
        dx = layer.backward(dout)

        assert dx.shape == X.shape

    def test_avg_pooling_backward_values(self):
        # Gradient distributed equally
        X = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        layer = AveragePooling2D(pool_size=2, strides=2)
        layer.build((1, 2, 2))

        out = layer.forward(X)
        dout = np.ones_like(out)  # grad 1.0
        dx = layer.backward(dout)

        # Each element contributed 1/4 to the mean
        expected_dx = np.full((1, 1, 2, 2), 0.25)

        assert np.allclose(dx, expected_dx)
