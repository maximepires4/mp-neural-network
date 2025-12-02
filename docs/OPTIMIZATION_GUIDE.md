# Optimization & Performance Guide

MPNeuralNetwork is designed to be performant, but achieving optimal speed requires understanding how to configure the framework correctly.

---

## 1. Data Precision (Float32)

**Rule #1: Use Float32.**

By default, Python and NumPy use `float64` (double precision). Deep Learning rarely benefits from this extra precision, but it costs **2x memory** and **~2x bandwidth**.

The framework enforces `float32` internally (`DTYPE = np.float32`), but you should ensure your input data is cast before feeding it to the model to avoid on-the-fly conversion overhead.

```python
# Default float64
X_train = np.random.randn(1000, 784)

# Explicit float32
X_train = np.random.randn(1000, 784).astype(np.float32)
```

---

## 2. Batch Size Selection

Choosing the right batch size is a trade-off between convergence stability and hardware utilization.

* **Too Small (1-16):** High overhead due to Python loops. The vectorization engine (BLAS/LAPACK) is starved of data.
* **Too Large (2048+):** Can lead to generalization issues and out-of-memory (OOM) errors.
* **Optimal (32-512):** Typically, powers of 2 like **32, 64, 128, or 256** provide the best balance.

---

## 3. Hardware Acceleration (GPU)

MPNN supports NVIDIA GPUs via **CuPy**. This provides massive speedups for large matrix multiplications (Dense layers) and Convolutions.

### Prerequisites

* NVIDIA GPU
* CUDA Toolkit installed
* `cupy` python package installed (`pip install cupy-cuda11x` or similar)

### Enabling GPU Mode

Set the environment variable `MPNN_BACKEND` before running your script.

```bash
# Run on GPU
export MPNN_BACKEND=cupy
python my_script.py
```

Or inside Python (before importing `mpneuralnetwork`):

```python
import os
os.environ["MPNN_BACKEND"] = "cupy"
import mpneuralnetwork
```

---

## 4. Benchmarking

The project includes a robust benchmarking suite to measure performance improvements or regressions.

### Running Benchmarks

Benchmarks are located in the `benchmark/` directory. The runner script executes them and profiles both time and memory.

```bash
python benchmark/run_benchmarks.py
```

This will generate reports in `output/benchmark_TIMESTAMP/`:

* `*.prof`: CPU profile data.
* `*.bin`: Memory usage data (Memray).
* `flamegraph.html`: Interactive memory usage visualization.

### Analyzing Results

Use `snakeviz` to visualize CPU bottlenecks:

```bash
snakeviz output/benchmark_.../cpu_profile.prof
```

---

## 5. Common Bottlenecks

### `im2col` Memory Usage

Convolutional layers use `im2col` to vectorize operations. This expands the input image into a large matrix.

* **Impact:** Memory usage grows by factor of $K^2$ (Kernel Size squared).
* **Mitigation:** If you run out of memory, try reducing the `batch_size` or using smaller kernels (e.g., 3x3 instead of 5x5).

### Data Copying

The framework tries to minimize copies, but some operations (like `flatten` or `transpose` on non-contiguous arrays) force a copy.

* **Tip:** Ensure your data is C-contiguous if you are doing manual pre-processing: `x = np.ascontiguousarray(x)`.
