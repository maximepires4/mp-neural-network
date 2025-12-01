<p align="center">
  <img src="./images/logo.svg" alt="MPNeuralNetwork Logo" width="900"/>
</p>

# **MPNeuralNetwork 🧠**

[![PyPI version](https://img.shields.io/pypi/v/mpneuralnetwork.svg)](https://pypi.org/project/mpneuralnetwork/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-material-blue.svg)](https://maximepires4.github.io/mp-neural-network/)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen?style=flat-square)

**A fully vectorized Deep Learning framework built from scratch using only NumPy.**

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
* **Early Stopping & Checkpointing:** Automatically monitors validation loss and restores the best weights.
* **Smart Initialization:** Automatically applies **He Init** (for ReLU) or **Xavier** (for Sigmoid/Tanh).
* **Comprehensive Regularization:** Supports **Dropout**, **L1/L2 Weight Decay** (AdamW style).
* **Numerical Stability:** Internally handles logits for Softmax/Sigmoid to prevent overflow.
* **Full Serialization:** Save/Load model state to `.npz` files.

[**👉 Learn more about the internal engine**](docs/INTERNALS.md)

## **Installation**

```bash
pip install mpneuralnetwork
```

## **Quick Start**

### **MNIST Classification**

```python
from mpneuralnetwork.layers import Dense, Dropout
from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.optimizers import Adam
from mpneuralnetwork.model import Model

# 1. Define the Architecture
network = [
    Dense(128, input_size=784), # Auto-He Init
    ReLU(),
    Dropout(0.2),
    Dense(10)                   # Output Logits
]

# 2. Initialize
model = Model(
    layers=network,
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001)
)

# 3. Train (Auto-Validation Split)
model.train(X_train, y_train, epochs=10, batch_size=32, auto_evaluation=0.2)
```

[**👉 See full tutorials in the User Guide**](docs/USER_GUIDE.md)

## **Architecture & Performance**

### **Vectorization**

The training loop handles 3D/2D tensors, replacing slow Python loops with NumPy's BLAS routines. Convolutional layers use the **`im2col`** technique, transforming convolutions into efficient Matrix Multiplications (GEMM).

### **Optimization**

The framework enforces **Float32** precision globally to halve memory usage and double bandwidth. Recent benchmarks show a **26% speedup** and **50% memory reduction** compared to the initial implementation.

[**👉 Read the Optimization & Benchmarking Guide**](docs/OPTIMIZATION_GUIDE.md)

## **Roadmap**

[**👉 View the full Roadmap**](docs/ROADMAP.md)

## **Author**

**Maxime Pires** - *AI Engineer | CentraleSupelec*

[LinkedIn](https://www.linkedin.com/in/maximepires) | [Portfolio](https://github.com/maximepires4)
