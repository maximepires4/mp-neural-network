# CHANGELOG

<!-- version list -->

## v1.2.0-beta.1 (2025-12-14)

### Features

- Add an option to disable bias correction on Adam
  ([`a4bc8e9`](https://github.com/maximepires4/mp-neural-network/commit/a4bc8e9eb357b856e39cefa8dd7af477963b0184))


## v1.1.1 (2025-12-03)

### Bug Fixes

- Fixed outdated documentation
  ([`2eb10d4`](https://github.com/maximepires4/mp-neural-network/commit/2eb10d42acc377bb0c09cff383eaa1ab307d088d))


## v1.1.0 (2025-12-02)

### Features

- Add documentation with mkdocs
  ([`f4f0273`](https://github.com/maximepires4/mp-neural-network/commit/f4f027382dc3ae76b52e657ab90d74dbd39b2669))

- Add GPU support with CuPy
  ([`fa4da85`](https://github.com/maximepires4/mp-neural-network/commit/fa4da85e1f2b3ffffb2bcaca27e6e9eeeaf5cab2))

- Add Softmax temperature
  ([`cb19a95`](https://github.com/maximepires4/mp-neural-network/commit/cb19a95e387ad80804315c90353f56fc4fbb5334))

- Add stride and padding for Convolutional layer
  ([`4954688`](https://github.com/maximepires4/mp-neural-network/commit/495468821f93ac883887995bab20fb144772daef))

### Refactoring

- Added deprecated functions for avoiding breaking change
  ([`e6c6f02`](https://github.com/maximepires4/mp-neural-network/commit/e6c6f02de9924a5f1705b1ed3723de0f8c5bcd73))

### Testing

- Refactor tests, GPU tests
  ([`02950b1`](https://github.com/maximepires4/mp-neural-network/commit/02950b1da176f220ecac8027e1e0b6a521a08927))

## v1.0.0 (2025-11-28)

### Features

- **Layers:**
    - `Dense` (fully connected) with `no_bias` support.
    - `Convolutional` (Conv2D) using `im2col` for efficiency.
    - `MaxPooling2D` and `AveragePooling2D`.
    - `BatchNormalization` (supports both 1D Dense and 2D Spatial/CNN).
    - `Dropout` for regularization.
    - `Flatten`.
- **Optimizers:**
    - `SGD` (with Momentum).
    - `RMSprop`.
    - `Adam` (with decoupled weight decay support).
    - **Regularization:** L1 and L2 weight decay integrated into all optimizers.
- **Activations:** `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `PReLU`, `Swish`.
- **Losses & Metrics:**
    - `MSE` (Mean Squared Error).
    - `BinaryCrossEntropy`.
    - `CategoricalCrossEntropy` (with numerical stability fixes for logits).
    - Standalone metrics system including `Accuracy`, `Precision`, `Recall`, `F1Score`, `RMSE`, `MAE`, `R2Score`.
- **Training Engine:**
    - **Early Stopping:** Monitors validation loss to stop training when improvement stalls.
    - **Model Checkpoint:** Automatically saves the best model weights.
    - **Auto-Validation:** Automatic splitting of training data for validation.
    - **Smart Defaults:** Automatic weight initialization (He/Xavier) and shape inference (no need to specify input shapes for every layer).
- **Serialization:** Complete `save_model` and `load_model` support (architecture + weights) using `.json` and compressed `.npz`.

### Performance

- **Vectorization:** Full batch vectorization for all layers and backpropagation.
- **Float32:** Enforced global `np.float32` dtype (vs default float64), reducing RAM usage by ~50% and speeding up operations by ~20%.
- **In-Place Operations:** optimized training and evaluation loops to eliminate redundant array copies (~13% speedup).
- **Optimized Shuffling:** Shuffling indices instead of physical arrays during training.
- **Lazy Metrics:** Metrics are computed only when necessary to reduce overhead.

### Infrastructure

- **Quality:**
    - Full static typing with `mypy`.
    - Linting and formatting with `ruff`.
    - CI/CD pipelines (GitHub Actions) for tests and publishing.
- **Benchmarking:** Dedicated scripts for performance profiling and regression testing.
- **Documentation:** Comprehensive README with examples (MNIST, Regression, CNN).
