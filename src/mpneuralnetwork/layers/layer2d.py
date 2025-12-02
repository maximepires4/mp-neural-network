from typing import Literal

from .. import DTYPE, ArrayType, xp
from .layer import Layer, Lit_W
from .utils import col2im, im2col


class Convolutional(Layer):
    """2D Convolutional Layer.

    Applies a 2D convolution over an input signal composed of several input planes.
    Uses the `im2col` optimization to convert convolution into matrix multiplication,
    allowing for efficient vectorization.

    Attributes:
        output_depth (int): Number of output channels (filters).
        kernel_size (int): Size of the square convolution kernel.
        stride (int): Step size of the convolution.
        padding (int): Amount of zero-padding applied to both sides of the input.
        initialization (Lit_W): Weight initialization strategy.
        no_bias (bool): Whether to disable bias.
        kernels (ArrayType): Learnable filters (output_depth, input_depth, k, k).
        biases (ArrayType): Learnable biases (output_depth,).
    """

    def __init__(
        self,
        output_depth: int,
        kernel_size: int,
        input_shape: tuple | None = None,
        initialization: Lit_W = "auto",
        no_bias: bool = False,
        padding: int | Literal["valid", "same"] = "valid",
        stride: int = 1,
    ) -> None:
        """Initializes the Convolutional layer.

        Args:
            output_depth (int): Number of filters.
            kernel_size (int): Height/Width of the filter (assumed square).
            input_shape (tuple | None, optional): Shape of input (depth, height, width).
            initialization (Lit_W, optional): Weight init method ("auto", "he", "xavier").
            no_bias (bool, optional): Disable bias. Defaults to False.
            padding (int | str, optional): Padding strategy. Can be an integer (amount of padding),
                "valid" (no padding), or "same" (padding to preserve spatial dimensions with stride=1).
                Defaults to "valid".
            stride (int, optional): Stride of the convolution. Defaults to 1.
        """
        super().__init__()
        self.output_depth: int = output_depth
        self.kernel_size: int = kernel_size
        self.initialization: Lit_W = initialization
        self.no_bias: bool = no_bias
        self.stride: int = stride
        self.padding_arg: int | Literal["valid", "same"] = padding

        self.padding: int
        if self.padding_arg == "valid":
            self.padding = 0
        elif self.padding_arg == "same":
            self.padding = (self.kernel_size - 1) // 2
        elif isinstance(self.padding_arg, int):
            self.padding = self.padding_arg
        else:
            raise ValueError("Padding must be 'valid', 'same', or an integer.")

        self.kernels: ArrayType
        self.kernels_gradient: ArrayType
        self.biases: ArrayType
        self.biases_gradient: ArrayType
        self.input_padded_shape: tuple

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
                "stride": self.stride,
                "padding": self.padding_arg,
            }
        )
        return config

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)

        _, input_height, input_width = self.input_shape

        output_height = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.output_shape = (self.output_depth, output_height, output_width)

        if self.initialization != "auto":
            self.init_weights(self.initialization, self.no_bias)

    def init_weights(self, method: Lit_W, no_bias: bool) -> None:
        """Initializes kernels and biases."""
        std_dev = 0.1

        input_depth, _, _ = self.input_shape

        if method == "he":
            std_dev = xp.sqrt(2.0 / (input_depth * self.kernel_size * self.kernel_size), dtype=DTYPE)
        elif method == "xavier":
            std_dev = xp.sqrt(1.0 / (input_depth * self.kernel_size * self.kernel_size), dtype=DTYPE)

        kernels_shape = (
            self.output_depth,
            input_depth,
            self.kernel_size,
            self.kernel_size,
        )

        self.kernels = xp.random.randn(*kernels_shape).astype(DTYPE) * std_dev
        self.kernels_gradient = xp.zeros_like(self.kernels, dtype=DTYPE)

        self.no_bias = no_bias

        if not self.no_bias:
            self.biases = xp.random.randn(self.output_depth).astype(DTYPE)
            self.biases_gradient = xp.zeros_like(self.biases, dtype=DTYPE)

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        """Performs 2D Convolution.

        Args:
            input_batch (ArrayType): Input data (N, C_in, H, W).
            training (bool, optional): Unused. Defaults to True.

        Returns:
            ArrayType: Feature maps (N, C_out, H_out, W_out).
        """
        self.input = input_batch

        if self.padding > 0:
            input_batch_padded = xp.pad(
                input_batch,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
            )
        else:
            input_batch_padded = input_batch

        self.input_padded_shape = input_batch_padded.shape

        input_windows = im2col(input_batch_padded, self.kernel_size, self.stride)

        output = xp.tensordot(input_windows, self.kernels, axes=((3, 4, 5), (1, 2, 3)))

        if not self.no_bias:
            output += self.biases

        return output.transpose(0, 3, 1, 2)  # type: ignore[no-any-return]

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        """Backpropagates gradients through convolution.

        Uses `col2im` to reconstruct the gradient for the input image.

        Args:
            output_gradient_batch (ArrayType): Gradient w.r.t output.

        Returns:
            ArrayType: Gradient w.r.t input.
        """
        grad_transposed = output_gradient_batch.transpose(0, 2, 3, 1)

        if not self.no_bias:
            self.biases_gradient = xp.sum(grad_transposed, axis=(0, 1, 2), dtype=DTYPE)

        if self.padding > 0:
            input_batch_padded = xp.pad(
                self.input,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
            )
        else:
            input_batch_padded = self.input

        input_windows = im2col(input_batch_padded, self.kernel_size, self.stride)

        self.kernels_gradient = xp.tensordot(grad_transposed, input_windows, axes=((0, 1, 2), (0, 1, 2)))

        input_grad_windows = xp.tensordot(grad_transposed, self.kernels, axes=((3), (0)))

        input_grad_windows_transposed = input_grad_windows.transpose(0, 3, 1, 2, 4, 5)

        input_grad_padded = col2im(
            input_grad_windows_transposed,
            self.input_padded_shape,
            self.output_shape,
            self.kernel_size,
            self.stride,
        )

        if self.padding > 0:
            input_grad = input_grad_padded[:, :, self.padding : -self.padding, self.padding : -self.padding]
        else:
            input_grad = input_grad_padded

        return input_grad

    @property
    def params(self) -> dict[str, tuple[ArrayType, ArrayType]]:
        return {
            "kernels": (self.kernels, self.kernels_gradient),
            "biases": (self.biases, self.biases_gradient),
        }

    def load_params(self, params: dict[str, ArrayType]) -> None:
        self.kernels[:] = params["kernels"]
        self.biases[:] = params["biases"]


class Flatten(Layer):
    """Flatten Layer.

    Flattens the input tensor into a 1D tensor (vector) per sample.
    Crucial for connecting Convolutional/Pooling layers to Dense layers.

    Input: (Batch, Channel, Height, Width)
    Output: (Batch, Channel * Height * Width)
    """

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        return input_batch.reshape(input_batch.shape[0], -1)

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        return output_gradient_batch.reshape(output_gradient_batch.shape[0], *self.input_shape)


class MaxPooling2D(Layer):
    """Max Pooling 2D Layer.

    Downsamples the input by taking the maximum value over a window.
    Reduces spatial dimensions and computation, while providing translational invariance.

    Attributes:
        pool_size (int): Size of the pooling window.
        stride (int): Stride of the pooling operation.
    """

    def __init__(self, pool_size: int = 2, strides: int | None = None):
        super().__init__()
        self.pool_size: int = pool_size
        self.stride: int = strides if strides is not None else pool_size

        self.windows: ArrayType

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)
        C, H, W = self.input_shape

        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        self.output_shape = (C, out_h, out_w)

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        self.input_shape = input_batch.shape
        self.windows = im2col(input_batch, self.pool_size, self.stride)
        max_val = xp.max(self.windows, axis=(4, 5))

        return max_val.transpose(0, 3, 1, 2)  # type: ignore

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        grad_transposed = output_gradient_batch.transpose(0, 2, 3, 1)

        grad_expanded = grad_transposed[..., None, None]

        max_vals = xp.max(self.windows, axis=(4, 5), keepdims=True)
        mask = self.windows == max_vals

        d_windows = grad_expanded * mask
        d_windows = d_windows.transpose(0, 3, 1, 2, 4, 5)

        output_shape_no_batch = output_gradient_batch.shape[1:]

        return col2im(
            d_windows,
            self.input_shape,
            output_shape_no_batch,
            self.pool_size,
            self.stride,
        )


class AveragePooling2D(Layer):
    """Average Pooling 2D Layer.

    Downsamples the input by taking the average value over a window.

    Attributes:
        pool_size (int): Size of the pooling window.
        stride (int): Stride of the pooling operation.
    """

    def __init__(self, pool_size: int = 2, strides: int | None = None):
        super().__init__()
        self.pool_size: int = pool_size
        self.stride: int = strides if strides is not None else pool_size

        self.windows: ArrayType

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)
        C, H, W = self.input_shape

        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        self.output_shape = (C, out_h, out_w)

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        self.input_shape = input_batch.shape
        self.windows = im2col(input_batch, self.pool_size, self.stride)
        means = xp.mean(self.windows, axis=(4, 5), dtype=DTYPE)

        return means.transpose(0, 3, 1, 2)  # type: ignore

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        grad_transposed = output_gradient_batch.transpose(0, 2, 3, 1)

        grad_expanded = grad_transposed[..., None, None]

        d_windows = grad_expanded * xp.ones_like(self.windows, dtype=DTYPE) / (self.pool_size * self.pool_size)
        d_windows = d_windows.transpose(0, 3, 1, 2, 4, 5)

        output_shape_no_batch = output_gradient_batch.shape[1:]

        return col2im(
            d_windows,
            self.input_shape,
            output_shape_no_batch,
            self.pool_size,
            self.stride,
        )


class BatchNormalization2D(Layer):
    """Batch Normalization Layer (2D) for Convolutional Networks.

    Normalize the activations of the previous layer at each batch.
    Operates on the channel dimension (axis 1), so statistics are computed
    over (Batch, Height, Width).

    Attributes:
        momentum (float): Momentum for the moving average.
        epsilon (float): Small float added to variance to avoid dividing by zero.
    """

    def __init__(self, momentum: float = 0.9, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.momentum: float = momentum
        self.epsilon: float = epsilon

        self.gamma: ArrayType
        self.beta: ArrayType

        self.cache_m: ArrayType
        self.cache_v: ArrayType

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        super().build(input_shape)
        C, H, W = self.input_shape

        self.gamma = xp.ones((1, C, 1, 1), dtype=DTYPE)
        self.gamma_gradient = xp.zeros_like(self.gamma, dtype=DTYPE)

        self.beta = xp.zeros((1, C, 1, 1), dtype=DTYPE)
        self.beta_gradient = xp.zeros_like(self.beta, dtype=DTYPE)

        self.cache_m = xp.zeros((1, C, 1, 1), dtype=DTYPE)
        self.cache_v = xp.ones((1, C, 1, 1), dtype=DTYPE)

        self.std_inv: ArrayType
        self.x_centered: ArrayType
        self.x_norm: ArrayType

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"momentum": self.momentum, "epsilon": self.epsilon})
        return config

    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        """Performs spatial batch normalization.

        Args:
            input_batch (ArrayType): Input (N, C, H, W).
            training (bool, optional): If True, updates running stats.

        Returns:
            ArrayType: Normalized input.
        """
        self.input = input_batch

        mean: ArrayType
        var: ArrayType

        if training:
            mean = xp.mean(self.input, axis=(0, 2, 3), keepdims=True, dtype=DTYPE)  # type: ignore
            var = xp.var(self.input, axis=(0, 2, 3), keepdims=True, dtype=DTYPE)

            self.cache_m = self.momentum * self.cache_m + (1 - self.momentum) * mean
            self.cache_v = self.momentum * self.cache_v + (1 - self.momentum) * var

        else:
            mean = self.cache_m
            var = self.cache_v

        self.std_inv = 1 / xp.sqrt(var + self.epsilon, dtype=DTYPE)
        self.x_centered = self.input - mean
        self.x_norm = self.x_centered * self.std_inv

        res: ArrayType = self.x_norm * self.gamma
        res += self.beta
        return res

    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        self.gamma_gradient = xp.sum(  # type: ignore
            self.x_norm * output_gradient_batch,
            axis=(0, 2, 3),
            keepdims=True,
            dtype=DTYPE,
        )
        self.beta_gradient = xp.sum(output_gradient_batch, axis=(0, 2, 3), keepdims=True, dtype=DTYPE)  # type: ignore

        N = output_gradient_batch.shape[0] * output_gradient_batch.shape[2] * output_gradient_batch.shape[3]

        dx_norm = output_gradient_batch * self.gamma

        grad: ArrayType = (
            (1 / N)
            * self.std_inv
            * (
                N * dx_norm
                - xp.sum(dx_norm, axis=(0, 2, 3), keepdims=True, dtype=DTYPE)
                - self.x_norm * xp.sum(dx_norm * self.x_norm, axis=(0, 2, 3), keepdims=True, dtype=DTYPE)
            )
        )
        return grad

    @property
    def params(self) -> dict[str, tuple[ArrayType, ArrayType]]:
        return {  # type: ignore
            "gamma": (self.gamma, self.gamma_gradient),
            "beta": (self.beta, self.beta_gradient),
        }

    def load_params(self, params: dict[str, ArrayType]) -> None:
        self.gamma[:] = params["gamma"]
        self.beta[:] = params["beta"]

    @property
    def state(self) -> dict[str, ArrayType]:
        return {"cache_m": self.cache_m, "cache_v": self.cache_v}

    @state.setter
    def state(self, state: dict[str, ArrayType]) -> None:
        self.cache_m = state["cache_m"]
        self.cache_v = state["cache_v"]
