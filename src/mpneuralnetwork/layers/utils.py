from .. import DTYPE, ArrayType, xp


def im2col(input_batch: ArrayType, window_size: int, stride: int | None = None) -> ArrayType:
    """Image to Column transformation.

    Rearranges image blocks into columns to perform convolution as a matrix multiplication.
    This is the core optimization that enables vectorized convolution.

    Transformation:
        Input:  (N, C, H, W)
        Output: (N, H_out, W_out, C * K * K) (before flattening for matmul)

    Args:
        input_batch (ArrayType): Input images of shape (N, C, H, W).
        window_size (int): Size of the kernel (K).
        stride (int | None, optional): Stride of the operation. Defaults to window_size if None (for pooling) or 1.

    Returns:
        ArrayType: The column matrix ready for GEMM (General Matrix Multiplication).
    """
    windows = xp.lib.stride_tricks.sliding_window_view(input_batch, window_shape=(window_size, window_size), axis=(2, 3))  # type: ignore[call-overload]

    if stride is not None:
        windows = windows[:, :, ::stride, ::stride, :, :]

    return windows.transpose(0, 2, 3, 1, 4, 5)  # type: ignore[no-any-return]


def col2im(
    cols: ArrayType,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    window_size: int,
    stride: int = 1,
) -> ArrayType:
    """Column to Image transformation (Reverse im2col).

    Used during backpropagation to reconstruct the gradient of the input image
    from the gradients of the columns. Accumulates gradients in overlapping regions.

    Args:
        cols (ArrayType): The column matrix from the gradient calculation.
        input_shape (tuple[int, ...]): Original shape of the input image (N, C, H, W).
        output_shape (tuple[int, ...]): Shape of the output (N, C_out, H_out, W_out).
        window_size (int): Kernel size.
        stride (int, optional): Stride. Defaults to 1.

    Returns:
        ArrayType: Reconstructed image gradient of shape (N, C, H, W).
    """
    _, H_out, W_out = output_shape
    K = window_size

    im = xp.zeros(input_shape, dtype=DTYPE)

    for i in range(K):
        for j in range(K):
            im[:, :, i : i + H_out * stride : stride, j : j + W_out * stride : stride] += cols[:, :, :, :, i, j]

    return im
