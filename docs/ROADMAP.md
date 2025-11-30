# Project Roadmap

This roadmap outlines the planned improvements and features for `mp-neural-network`.

## Upcoming features (v1.1)

- [x] **GPU Acceleration:** Explore optional CuPy backend for NVIDIA GPU support.
- [ ] **Stable parameter identification system**: Remove dependency on `id(param)` for tracking optimizer states (moments/velocities), implement a more robust approach (e.g., persistent UUIDs or structured naming).
- [ ] **Explicit Shape Checking:** Add clear error messages when connecting incompatible layers (e.g., "Dense(10) -> Dense(50): Mismatch").
- [ ] **Advanced Documentation:** Add Google-style docstrings to all public classes, exposing hidden features like "Smart Weights" and "He Initialization".
- [ ] **CNN Padding and strides**: Support for padding and strides in convolutional layers.

## Future / Exploration

- [ ] **RNN/LSTM Layers:** Support for sequential data.
- [ ] **Advanced Schedulers:** Implement Learning Rate Schedulers (ReduceLROnPlateau, CosineAnnealing).
- [ ] **Data Loaders:** Create a generator-based `DataLoader` for training on datasets larger than RAM (or VRAM).

## Completed Features (v1.0+)

- [x] **Float32 Transition:** Global `DTYPE = np.float32` enforced to reduce RAM by ~50%.
- [x] **In-Place Operations:** Eliminated redundant copies in `train`, `predict`, and `evaluate`.
- [x] **Optimized Shuffle:** Shuffling indices instead of copying the full dataset.
- [x] **Lazy Metrics:** Metrics are computed only when necessary, speeding up training.
- [x] **Batch Vectorization**
- [x] **Numerical Stability Fixes (Logits)**
- [x] **Advanced Optimizers:** Adam, RMSprop, SGD Momentum.
- [x] **Smart Initialization:** Auto He/Xavier.
- [x] **Regularization:** Dropout Layer & L1/L2 Weight Decay.
- [x] **Convolutional Layers:** Conv2D implementation with `im2col`.
- [x] **Model Serialization:** Saving/Loading weights to JSON/Pickle.
- [x] **Training Utils:** Early Stopping, Checkpointing, Auto-Metrics.
- [x] **Pooling Layers:** MaxPool / AvgPool.
- [x] **BatchNormalization:** 1D (Dense) and 2D (Spatial/CNN).
- [x] **Memory Optimization:** In-place operations, reduced copies, global float32.
- [x] **Performance Boost:** Training loop optimization (index shuffling) and type enforcement (~26% speedup).
