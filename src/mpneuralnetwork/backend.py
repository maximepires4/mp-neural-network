import os
from types import ModuleType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp

    _cupy_available = True
except ImportError:
    cp = None
    _cupy_available = False

BACKEND_TYPE = os.getenv("MPNN_BACKEND", "numpy").lower()

xp: ModuleType

if BACKEND_TYPE == "cupy" and _cupy_available and cp is not None:
    xp = cp
    DTYPE = cp.float32
    print("Backend: CuPy (GPU)")
else:
    if BACKEND_TYPE == "cupy":
        print("CuPy not found. Fallback on NumPy (CPU).")
    xp = np
    DTYPE = np.float32

ArrayType: TypeAlias = np.ndarray | Any
"""TypeAlias: Union[np.ndarray, cp.ndarray] - Represents an array on either CPU or GPU."""


def to_device(array: ArrayType) -> ArrayType:
    """Transfers an array to the current backend device.

    If the backend is CPU (NumPy), returns the array as a numpy array.
    If the backend is GPU (CuPy), moves the array to GPU memory.

    Args:
        array (ArrayType): The input array (numpy or cupy).

    Returns:
        ArrayType: The array on the configured device.
    """
    if xp.__name__ == "cupy":
        return xp.asarray(array)
    return np.asarray(array)


def to_host(array: ArrayType) -> NDArray:
    """Transfers an array to the CPU (Host).

    If the array is on GPU (CuPy), it is transferred to CPU.
    If it's already on CPU, it is returned as is.

    Args:
        array (ArrayType): The input array.

    Returns:
        NDArray: A NumPy array.
    """
    if hasattr(array, "get"):
        return array.get()  # type: ignore
    return np.asarray(array)


def get_backend() -> ModuleType:
    """Returns the current backend module.

    Returns:
        ModuleType: `numpy` or `cupy` module.
    """
    return xp
