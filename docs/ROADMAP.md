# Project Roadmap

This roadmap outlines the planned improvements and features for `mp-neural-network`.

## Upcoming features (v1.1)

- **GPU Acceleration:** Explore optional CuPy backend for NVIDIA GPU support.
- **Advanced Documentation:** Add Google-style docstrings to all public classes
- **CNN Padding and strides**: Support for padding and strides in convolutional layers.
- **Softmax temperature:** Support for temperature scaling in softmax.

## Future / Exploration

- **RNN/LSTM Layers:** Support for sequential data.
- **Channels last optimization:** Optimize convolutional layers for channels_last format.
- **Advanced Schedulers:** Implement Learning Rate Schedulers (ReduceLROnPlateau, CosineAnnealing).
- **Data Loaders:** Create a generator-based `DataLoader` for training on datasets larger than RAM (or VRAM).
- **Stable parameter identification system**: Remove dependency on `id(param)` for tracking optimizer states (moments/velocities), implement a more robust approach (e.g., persistent UUIDs or structured naming).
- **Explicit Shape Checking:** Add clear error messages when connecting incompatible layers

## Completed Features (v1.0+)

- **Float32 Transition:** Global `DTYPE = np.float32` enforced to reduce RAM by ~50%.
- **In-Place Operations:** Eliminated redundant copies in `train`, `predict`, and `evaluate`.
- **Optimized Shuffle:** Shuffling indices instead of copying the full dataset.
- **Lazy Metrics:** Metrics are computed only when necessary, speeding up training.
- **Batch Vectorization**
- **Numerical Stability Fixes (Logits)**
- **Advanced Optimizers:** Adam, RMSprop, SGD Momentum.
- **Smart Initialization:** Auto He/Xavier.
- **Regularization:** Dropout Layer & L1/L2 Weight Decay.
- **Convolutional Layers:** Conv2D implementation with `im2col`.
- **Model Serialization:** Saving/Loading weights to JSON/Pickle.
- **Training Utils:** Early Stopping, Checkpointing, Auto-Metrics.
- **Pooling Layers:** MaxPool / AvgPool.
- **BatchNormalization:** 1D (Dense) and 2D (Spatial/CNN).
- **Memory Optimization:** In-place operations, reduced copies, global float32.
- **Performance Boost:** Training loop optimization (index shuffling) and type enforcement (~26% speedup).
