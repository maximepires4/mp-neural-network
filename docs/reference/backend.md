# Backend & Hardware Acceleration

MPNeuralNetwork supports both CPU and GPU execution through a unified backend interface. This module handles the abstraction between `numpy` (CPU) and `cupy` (GPU).

## Configuration

The backend is selected at runtime via the `MPNN_BACKEND` environment variable.

- **CPU (Default):** `MPNN_BACKEND=numpy`
- **GPU (NVIDIA):** `MPNN_BACKEND=cupy` (Requires `cupy` to be installed)

## API Reference

### Global Types

::: mpneuralnetwork.backend.DTYPE
    options:
        show_source: false

::: mpneuralnetwork.backend.ArrayType
    options:
        show_source: false

### Functions

::: mpneuralnetwork.backend.to_device
::: mpneuralnetwork.backend.to_host
::: mpneuralnetwork.backend.get_backend
