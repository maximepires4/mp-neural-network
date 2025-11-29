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


def to_device(array: ArrayType) -> ArrayType:
    if xp.__name__ == "cupy":
        return xp.asarray(array)
    return np.asarray(array)


def to_host(array: ArrayType) -> NDArray:
    if hasattr(array, "get"):
        return array.get()  # type: ignore
    return np.asarray(array)


def get_backend() -> ModuleType:
    return xp
