from .. import DTYPE, ArrayType, xp


def im2col(input_batch: ArrayType, window_size: int, stride: int | None = None) -> ArrayType:
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
    _, H_out, W_out = output_shape
    K = window_size

    im = xp.zeros(input_shape, dtype=DTYPE)

    for i in range(K):
        for j in range(K):
            im[:, :, i : i + H_out * stride : stride, j : j + W_out * stride : stride] += cols[:, :, :, :, i, j]

    return im
