<p align="center">
  <img src="https://raw.githubusercontent.com/maximepires4/mp-neural-network/main/docs/assets/logo-full.svg" alt="MPNeuralNetwork Logo" width="900"/>
</p>

# **MPNeuralNetwork 🧠**

[![PyPI version](https://img.shields.io/pypi/v/mpneuralnetwork.svg)](https://pypi.org/project/mpneuralnetwork/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-material-blue.svg)](https://maximepires4.github.io/mp-neural-network/)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen?style=flat-square)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://github.com/numpy/numpy)
[![CuPy](https://img.shields.io/badge/cupy-%2376B900.svg?style=flat&logo=nvidia&logoColor=white)](https://github.com/cupy/cupy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

**A fully vectorized Deep Learning framework built from scratch using only [NumPy](https://github.com/numpy/numpy) and [CuPy](https://github.com/cupy/cupy).**

[**📖 Read the Full Documentation**](https://maximepires4.github.io/mp-neural-network/)

## **Philosophy & Goal**

In an era of high-level frameworks like PyTorch or TensorFlow, it is easy to treat Neural Networks as "black boxes".

**MPNeuralNetwork** is an engineering initiative designed to demystify the underlying mathematics of Deep Learning. By rebuilding the engine from the ground up, I aimed to bridge the gap between theoretical equations and production-grade code.

## **Key Objectives:**

1. **Mathematical Rigor:** Implementing backpropagation, chain rule derivatives, and loss functions manually.
2. **Performance Optimization:** Moving from naive scalar loops to **fully vectorized matrix operations** and implementing **`im2col`** for convolutions.
3. **Software Architecture:** Applying **SOLID principles** for a modular design.

## **Key Features**

MPNeuralNetwork goes beyond basic matrix operations by incorporating an **"intelligent" engine**.

* **Fully Vectorized:** Optimized for batch processing. Convolutions use **`im2col`** for hardware acceleration.
* **GPU Acceleration:** Seamless support for NVIDIA GPUs via **CuPy**. Switch backends with a single environment variable.
* **Early Stopping & Checkpointing:** Automatically monitors validation loss and restores the best weights.
* **Smart Initialization:** Automatically applies **He Init** (for ReLU) or **Xavier** (for Sigmoid/Tanh).
* **Comprehensive Regularization:** Supports **Dropout**, **L1/L2 Weight Decay** (AdamW style).
* **Numerical Stability:** Internally handles logits for Softmax/Sigmoid to prevent overflow.
* **Full Serialization:** Save/Load model state to `.npz` files.

[**👉 Learn more about the internal engine**](https://maximepires4.github.io/mp-neural-network/INTERNALS)

## **Component Inventory**

| Category | Available Components |
| :--- | :--- |
| **Layers** | `Dense`, `Convolutional`, `MaxPooling2D`, `AveragePooling2D`, `Dropout`, `BatchNormalization`, `Flatten` |
| **Activations** | `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `PReLU`, `Swish` |
| **Optimizers** | `SGD`, `RMSprop`, `AdamW` |
| **Losses** | `MSE`, `BinaryCrossEntropy`, `CategoricalCrossEntropy` |

## **Installation**

```bash
pip install mpneuralnetwork
```

## **Quick Start**

### **MNIST Classification**

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from mpneuralnetwork.layers import Dense, Dropout
from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.optimizers import Adam
from mpneuralnetwork.model import Model

# 1. Load Data (MNIST)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = (X / 255.0).astype(np.float32) # Normalize & Float32
y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

# 2. Define the Architecture
network = [
    Dense(128, input_size=784), # Auto-He Init
    ReLU(),
    Dropout(0.2),
    Dense(10)                   # Output Logits
]

# 3. Initialize
model = Model(
    layers=network,
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001)
)

# 4. Train (Auto-Validation Split)
model.train(X, y, epochs=5, batch_size=64, auto_evaluation=0.2)
```

[**👉 See full tutorials in the User Guide**](https://maximepires4.github.io/mp-neural-network/USER_GUIDE)

## **Architecture & Performance**

### **Vectorization**

The training loop handles 3D/2D tensors, replacing slow Python loops with NumPy's BLAS routines. Convolutional layers use the **`im2col`** technique, transforming convolutions into efficient Matrix Multiplications (GEMM).

### **Optimization**

The framework enforces **Float32** precision globally to halve memory usage and double bandwidth. Recent benchmarks show a **26% speedup** and **50% memory reduction** compared to the initial implementation.

[**👉 Read the Optimization & Benchmarking Guide**](https://maximepires4.github.io/mp-neural-network/OPTIMIZATION_GUIDE)

## **Roadmap**

[**👉 View the full Roadmap**](https://maximepires4.github.io/mp-neural-network/ROADMAP)

## **Author**

**Maxime Pires** - *AI Engineer | CentraleSupelec*

[LinkedIn](https://www.linkedin.com/in/maximepires) | [Portfolio](https://github.com/maximepires4)
