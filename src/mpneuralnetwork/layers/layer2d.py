import numpy as np
from numpy.typing import NDArray

from .layer import Layer, Lit_W
from .utils import col2im, im2col


class Convolutional(Layer):
    def __init__(
        self, output_depth: int, kernel_size: int, input_shape: tuple | None = None, initialization: Lit_W = "auto", no_bias: bool = False
    ) -> None:
        super().__init__()
        self.output_depth: int = output_depth
        self.kernel_size: int = kernel_size
        self.initialization: Lit_W = initialization
        self.no_bias: bool = no_bias

        self.X_col: NDArray
        self.kernels: NDArray
        self.kernels_gradient: NDArray
        self.biases: NDArray
        self.biases_gradient: NDArray

        if input_shape is not None:
            self.build(input_shape)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "output_depth": self.output_depth,
                "kernel_size": self.kernel_size,
                "input_shape": self.input_shape,
                "initialization": self.initialization,
                "no_bias": self.no_bias,
            }
        )
        return config

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)

        _, input_height, input_width = self.input_shape

        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1
        self.output_shape = (self.output_depth, output_height, output_width)
        self.output_shape = self.output_shape

        if self.initialization != "auto":
            self.init_weights(self.initialization, self.no_bias)

    def init_weights(self, method: Lit_W, no_bias: bool) -> None:
        std_dev = 0.1

        input_depth, _, _ = self.input_shape

        if method == "he":
            std_dev = np.sqrt(2.0 / (input_depth * self.kernel_size * self.kernel_size))
        elif method == "xavier":
            std_dev = np.sqrt(1.0 / (input_depth * self.kernel_size * self.kernel_size))

        kernels_shape = (self.output_depth, input_depth, self.kernel_size, self.kernel_size)

        self.kernels = np.random.randn(*kernels_shape) * std_dev
        self.kernels_gradient = np.zeros_like(self.kernels)

        self.no_bias = no_bias

        if not self.no_bias:
            self.biases = np.random.randn(self.output_depth)
            self.biases_gradient = np.zeros_like(self.biases)

    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        self.input = input_batch
        batch_size = input_batch.shape[0]

        self.X_col = im2col(input_batch, self.kernel_size).reshape(-1, input_batch.shape[1] * self.kernel_size * self.kernel_size)

        k_col = self.kernels.reshape(self.output_depth, -1)

        _, output_height, output_width = self.output_shape

        output = self.X_col @ k_col.T
        if not self.no_bias:
            output += self.biases
        output = output.reshape(batch_size, output_height, output_width, self.output_depth)
        return output.transpose(0, 3, 1, 2)  # type: ignore[no-any-return]

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        dOut_col = output_gradient_batch.transpose(0, 2, 3, 1).reshape(-1, self.output_depth)

        if not self.no_bias:
            self.biases_gradient = np.sum(dOut_col, axis=0)

        k_col = dOut_col.T @ self.X_col
        self.kernels_gradient = k_col.reshape(self.output_depth, self.input_shape[0], self.kernel_size, self.kernel_size)

        W_col = self.kernels.reshape(self.output_depth, -1)
        dX_col = dOut_col @ W_col

        batch_size = output_gradient_batch.shape[0]
        dX_col = dX_col.reshape(
            batch_size, self.output_shape[1], self.output_shape[2], self.input_shape[0], self.kernel_size, self.kernel_size
        ).transpose(0, 3, 1, 2, 4, 5)

        return col2im(dX_col, self.input.shape, self.output_shape, self.kernel_size)

    @property
    def params(self) -> dict[str, tuple[NDArray, NDArray]]:
        return {"kernels": (self.kernels, self.kernels_gradient), "biases": (self.biases, self.biases_gradient)}

    def load_params(self, params: dict[str, NDArray]) -> None:
        self.kernels[:] = params["kernels"]
        self.biases[:] = params["biases"]


class Flatten(Layer):
    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        return input_batch.reshape(input_batch.shape[0], -1)

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        return output_gradient_batch.reshape(output_gradient_batch.shape[0], *self.input_shape)


class MaxPooling2D(Layer):
    def __init__(self, pool_size: int = 2, strides: int | None = None):
        super().__init__()
        self.pool_size: int = pool_size
        self.stride: int = strides if strides is not None else pool_size

        self.windows: NDArray

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)
        C, H, W = self.input_shape

        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        self.output_shape = (C, out_h, out_w)

    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        self.input_shape = input_batch.shape
        self.windows = im2col(input_batch, self.pool_size, self.stride)
        max_val = np.max(self.windows, axis=(4, 5))

        return max_val.transpose(0, 3, 1, 2)  # type: ignore[no-any-return]

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        grad_transposed = output_gradient_batch.transpose(0, 2, 3, 1)

        grad_expanded = grad_transposed[..., None, None]

        max_vals = np.max(self.windows, axis=(4, 5), keepdims=True)
        mask = self.windows == max_vals

        d_windows = grad_expanded * mask
        d_windows = d_windows.transpose(0, 3, 1, 2, 4, 5)

        output_shape_no_batch = output_gradient_batch.shape[1:]

        return col2im(d_windows, self.input_shape, output_shape_no_batch, self.pool_size, self.stride)


class AveragePooling2D(Layer):
    def __init__(self, pool_size: int = 2, strides: int | None = None):
        super().__init__()
        self.pool_size: int = pool_size
        self.stride: int = strides if strides is not None else pool_size

        self.windows: NDArray

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)
        C, H, W = self.input_shape

        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        self.output_shape = (C, out_h, out_w)

    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        self.input_shape = input_batch.shape
        self.windows = im2col(input_batch, self.pool_size, self.stride)
        means = np.mean(self.windows, axis=(4, 5))

        return means.transpose(0, 3, 1, 2)  # type: ignore[no-any-return]

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        grad_transposed = output_gradient_batch.transpose(0, 2, 3, 1)

        grad_expanded = grad_transposed[..., None, None]

        d_windows = grad_expanded * np.ones_like(self.windows) / (self.pool_size * self.pool_size)
        d_windows = d_windows.transpose(0, 3, 1, 2, 4, 5)

        output_shape_no_batch = output_gradient_batch.shape[1:]

        return col2im(d_windows, self.input_shape, output_shape_no_batch, self.pool_size, self.stride)


class BatchNormalization2D(Layer):
    def __init__(self, momentum: float = 0.9, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.momentum: float = momentum
        self.epsilon: float = epsilon

        self.gamma: NDArray
        self.beta: NDArray

        self.cache_m: NDArray
        self.cache_v: NDArray

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)
        C, H, W = self.input_shape

        self.gamma = np.ones((1, C, 1, 1))
        self.gamma_gradient = np.zeros_like(self.gamma)

        self.beta = np.zeros((1, C, 1, 1))
        self.beta_gradient = np.zeros_like(self.beta)

        self.cache_m = np.zeros((1, C, 1, 1))
        self.cache_v = np.ones((1, C, 1, 1))

        self.std_inv: NDArray
        self.x_centered: NDArray
        self.x_norm: NDArray

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"momentum": self.momentum, "epsilon": self.epsilon})
        return config

    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        self.input = input_batch

        mean: NDArray
        var: NDArray

        if training:
            mean = np.mean(self.input, axis=(0, 2, 3), keepdims=True)
            var = np.var(self.input, axis=(0, 2, 3), keepdims=True)

            self.cache_m = self.momentum * self.cache_m + (1 - self.momentum) * mean
            self.cache_v = self.momentum * self.cache_v + (1 - self.momentum) * var

        else:
            mean = self.cache_m
            var = self.cache_v

        self.std_inv = 1 / np.sqrt(var + self.epsilon)
        self.x_centered = self.input - mean
        self.x_norm = self.x_centered * self.std_inv

        res: NDArray = self.x_norm * self.gamma + self.beta
        return res

    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        self.gamma_gradient = np.sum(self.x_norm * output_gradient_batch, axis=(0, 2, 3), keepdims=True)
        self.beta_gradient = np.sum(output_gradient_batch, axis=(0, 2, 3), keepdims=True)

        N = output_gradient_batch.shape[0] * output_gradient_batch.shape[2] * output_gradient_batch.shape[3]

        dx_norm = output_gradient_batch * self.gamma

        grad: NDArray = (
            (1 / N)
            * self.std_inv
            * (
                N * dx_norm
                - np.sum(dx_norm, axis=(0, 2, 3), keepdims=True)
                - self.x_norm * np.sum(dx_norm * self.x_norm, axis=(0, 2, 3), keepdims=True)
            )
        )
        return grad

    @property
    def params(self) -> dict[str, tuple[NDArray, NDArray]]:
        return {"gamma": (self.gamma, self.gamma_gradient), "beta": (self.beta, self.beta_gradient)}

    def load_params(self, params: dict[str, NDArray]) -> None:
        self.gamma[:] = params["gamma"]
        self.beta[:] = params["beta"]

    @property
    def state(self) -> dict[str, NDArray]:
        return {"cache_m": self.cache_m, "cache_v": self.cache_v}

    @state.setter
    def state(self, state: dict[str, NDArray]) -> None:
        self.cache_m = state["cache_m"]
        self.cache_v = state["cache_v"]
