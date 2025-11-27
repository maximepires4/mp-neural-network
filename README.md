<p align="center">
  <img src="https://raw.githubusercontent.com/maximepires4/mp-neural-network/main/images/logo.svg" alt="MPNeuralNetwork Logo" width="900"/>
</p>

# **MPNeuralNetwork 🧠**

[![PyPI version](https://img.shields.io/pypi/v/mpneuralnetwork.svg)](https://pypi.org/project/mpneuralnetwork/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen?style=flat-square)

**A fully vectorized Deep Learning framework built from scratch using only NumPy.**

## **Philosophy & Goal**

In an era of high-level frameworks like PyTorch, TensorFlow and Keras, it is easy to treat Neural Networks as "black boxes".

MPNeuralNetwork is an engineering initiative designed to demystify the underlying mathematics of Deep Learning.
By rebuilding the engine from the ground up, I aimed to bridge the gap between theoretical equations and production-grade code.

## **Key Objectives:**

1. **Mathematical Rigor:** Implementing backpropagation, chain rule derivatives, and loss functions manually.
2. **Performance Optimization:** Moving from naive scalar loops to **fully vectorized matrix operations** (Batch Processing) and implementing **`im2col`** for convolutions to significantly accelerate training times.
3. **Software Architecture:** Applying **SOLID principles** to decouple Layers, Optimizers, and Loss functions for a modular design.

## **Key Features: Smart & Efficient**

MPNeuralNetwork goes beyond basic matrix operations by incorporating an **"intelligent" engine** that automates Deep Learning best practices.

* **Fully Vectorized & Optimized:** Optimized for batch processing using NumPy broadcasting. Convolutions use **`im2col` vectorization**, transforming loops into matrix multiplications for hardware acceleration (BLAS/MKL).
* **Early Stopping & Checkpointing:** The training loop automatically monitors validation performance. It stops training early if the model stops learning and **automatically restores the best weights** found during training, ensuring you always get the most generalized model.
* **Intelligent Weight Initialization:** The model analyzes your network architecture (specifically the activation functions) and automatically applies the optimal initialization strategy (**He Initialization** for ReLU, **Xavier/Glorot** for Sigmoid/Tanh), removing the guesswork.
* **Comprehensive Regularization:** Supports **Dropout** layers as well as **L1 and L2 Weight Decay** integrated directly into all optimizers (SGD, Adam, RMSprop).
* **Numerical Stability (Auto-Logits):** The framework detects classification tasks and internally handles logits for `Softmax` or `Sigmoid`, preventing numerical overflow/underflow issues common in naive implementations.
* **Auto-Validation Split:** Simply pass `auto_evaluation=0.2` to automatically set aside 20% of your data for validation, without manual array slicing.
* **Full Serialization:** Save and load your entire model state (weights, architecture, optimizer momentum) to resume training later.

## **Implemented Components**

| Component | Details |
| :---- | :---- |
| **Layers** | `Dense`, `Convolutional` (Conv2D), `Dropout`, `Flatten`, `BatchNormalization` (1D & 2D), `MaxPooling2D`, `AveragePooling2D` |
| **Activations** | `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `PReLU`, `Swish` |
| **Loss Functions** | `MSE` (Regression), `BinaryCrossEntropy`, `CategoricalCrossEntropy` (Logits optimized) |
| **Metrics** | `RMSE`, `MAE`, `R2Score`, `Accuracy`, `Precision`, `Recall`, `F1Score`, `TopKAccuracy` |
| **Optimizers** | `SGD` (with Momentum), `RMSprop`, `Adam`, `AdamW`) |

## **Installation**

### **From PyPI**

You can install the package directly from PyPI:

```bash
pip install mpneuralnetwork
```

### **From source**

To experiment with the code or run the examples:

```bash
git clone https://github.com/maximepires4/mp-neural-network.git
cd mp-neural-network
pip install -e .
```

*Note: The only hard dependency is `numpy`. `pandas` is optional for running certain examples.*

## **Usage Examples**

### **1. Classic MNIST Classification**

The API is designed to be declarative and intuitive.

```python
import numpy as np
from mpneuralnetwork.layers import Dense, Dropout
from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.optimizers import Adam
from mpneuralnetwork.model import Model

# 1. Define the Architecture
network = [
    Dense(128, input_size=784), # Automatically uses He init
    ReLU(),
    Dropout(0.2),               # Regularization
    Dense(128, 10)              # Output Logits
]

# 2. Initialize the Model
model = Model(
    layers=network,
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001) # L2 Regularization is default
)

# 3. Train (Vectorized) with Auto-Evaluation
# - Splits 20% of data for validation
# - Stops if validation loss doesn't improve for 5 epochs
# - Saves the best model state automatically
model.train(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    auto_evaluation=0.2,
    early_stopping=5
)

# 4. Predict
# Returns probabilities (Softmax applied automatically)
predictions = model.predict(X_test)
```

### **2. Convolutional Neural Network**

Support for 2D Convolutions, Pooling, and Batch Normalization.

```python
from mpneuralnetwork.layers import Convolutional, Flatten, Dense, MaxPooling2D, BatchNormalization2D
from mpneuralnetwork.activations import ReLU

cnn_network = [
    # Input: (Batch, 1, 28, 28) -> Output: (Batch, 32, 26, 26)
    Convolutional(input_shape=(1, 28, 28), output_depth=32, kernel_size=3),
    BatchNormalization2D(),
    ReLU(),

    # Pooling: (Batch, 32, 13, 13)
    MaxPooling2D(pool_size=2, strides=2),

    # Flatten: (Batch, 32 * 13 * 13)
    Flatten(),

    Dense(32 * 13 * 13, 100),
    ReLU(),
    Dense(100, 10)
]

model = Model(layers=cnn_network, ...)
```

### **3. Regression Example**

The framework supports regression tasks with MSE loss and relevant metrics.

```python
from mpneuralnetwork.losses import MSE
from mpneuralnetwork.metrics import RMSE, R2Score

# Architecture for regression
reg_network = [
    Dense(13, 64),
    ReLU(),
    Dense(64, 32),
    ReLU(),
    Dense(32, 1) # Linear output
]

model = Model(
    layers=reg_network,
    loss=MSE(),
    optimizer=Adam(learning_rate=0.01),
    metrics=[RMSE(), R2Score()] # Track multiple metrics
)

model.train(X_train, y_train, epochs=100, batch_size=16)
```

## **Architecture & Design Decisions**

### **1\. Vectorization & Performance**

Early versions of the library used loops to iterate over samples one by one. This was identified as a major bottleneck.

* **Refactoring:** I completely rewrote the main training loop (`Model.train`) and the forward/backward methods of all layers to handle 3D/2D tensors of shape `(batch_size, features)`.
* **Convolutional Optimization:** The `Convolutional` layer uses the **`im2col` (image-to-column)** technique. This transforms the sliding window convolution into a large matrix multiplication, allowing NumPy (and underlying BLAS libraries) to parallelize the operation efficiently.

### **2\. Decoupling Layers & Optimizers (SRP)**

To avoid "God Classes", I strictly separated the responsibility of **calculating gradients** from **updating parameters**. Layers expose their trainable parameters via a generic params property.

* **The Layer's Job:** It computes `dE/dW` (gradient) during the backward pass.
* **The Optimizer's Job:** The Optimizer class iterates over the layers, retrieves parameters via `layer.params`, and applies the update rule (keeping track of momentum/velocity if needed).

### **3. Unified Optimizer Design (Adam vs AdamW)**

A common confusion in Deep Learning libraries is the difference between **Adam + L2 Regularization** and **AdamW** (Decoupled Weight Decay).

* **Standard Approach:** Usually, L2 regularization is added to the loss function, meaning its gradient is added to the backpropagated gradient. In adaptive methods like Adam, this means the regularization term is effectively scaled by the inverse of the gradient variance, which can be suboptimal.
* **My Implementation:**
  * **L1 Regularization** follows the standard approach (added to gradients) to promote sparsity.
  * **L2 Regularization** is implemented as **Decoupled Weight Decay (AdamW)**. The decay is applied directly to the weights *after* the adaptive update step.

This design means that by simply setting `optimizer=Adam(regularization='L2')`, you are effectively using **AdamW**, the state-of-the-art optimizer for training modern deep networks.

## **Performance Benchmarks**

Optimization is at the core of this project. Recent benchmarks (v1.0.0b) show significant improvements compared to the initial implementation:

* **Speed (~26% faster):**
  * **Vectorization:** Replacing scalar loops with `im2col` for convolutions.
  * **In-Place Operations:** Using `out=...` in NumPy to avoid temporary array allocations in Optimizers and Layers.
  * **Smart Shuffling:** Shuffling indices instead of copying the entire dataset at every epoch.

* **Memory (~50% reduction):**
  * **Float32 Precision:** Enforced globally via `DTYPE` to halve the memory footprint of weights and gradients (vs default float64).
  * **Zero-Copy Views:** The training loop uses array views for validation splits and batching, eliminating redundant data duplication.

## **Benchmarking & Profiling**

To verify performance improvements (like `im2col` or `float32` optimization), the project includes a comprehensive benchmarking suite located in `benchmark/`.

### **Running Benchmarks**

A utility script automates the execution of all benchmarks, profiling both **CPU Time** (via `cProfile`) and **Memory Usage** (via `memray`).

```bash
# Run all benchmarks (generates .prof, .bin, and .html flamegraphs in output/benchmark_TIMESTAMP)
python benchmark/run_benchmarks.py
```

### **Comparing Performance**

You can compare two different runs (e.g., before and after an optimization) to see regression/improvements:

```bash
python benchmark/run_benchmarks.py --before output/benchmark_OLD --after output/benchmark_NEW
```

## **Roadmap**

* [x] **Batch Vectorization**
* [x] **Numerical Stability Fixes (Logits)**
* [x] **Advanced Optimizers:** Adam, RMSprop, SGD Momentum.
* [x] **Smart Initialization:** Auto He/Xavier.
* [x] **Regularization:** Dropout Layer & L1/L2 Weight Decay.
* [x] **Convolutional Layers:** Conv2D implementation with `im2col`.
* [x] **Model Serialization:** Saving/Loading weights to JSON/Pickle.
* [x] **Training Utils:** Early Stopping, Checkpointing, Auto-Metrics.
* [x] **Pooling Layers:** MaxPool / AvgPool.
* [x] **BatchNormalization:** 1D (Dense) and 2D (Spatial/CNN).
* [x] **Memory Optimization:** In-place operations, reduced copies, global float32.
* [x] **Performance Boost:** Training loop optimization (index shuffling) and type enforcement (~26% speedup).

## **Author**

**Maxime Pires** - *AI Engineer | CentraleSupelec*

Building robust AI systems by understanding the foundations.

[LinkedIn](https://www.linkedin.com/in/maximepires) | [Portfolio](https://github.com/maximepires4)
