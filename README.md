# MPNeuralNetwork 🧠

[![PyPI version](https://badge.fury.io/py/mpneuralnetwork.svg)](https://badge.fury.io/py/mpneuralnetwork)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)

**A fully vectorized Deep Learning framework built from scratch using only NumPy.**

-----

## Philosophy & Goal

In an era of high-level frameworks like PyTorch, TensorFlow and Keras, it is easy to treat Neural Networks as "black boxes".

**MPNeuralNetwork** is an engineering initiative designed to demystify the underlying mathematics of Deep Learning.
By rebuilding the engine from the ground up, I aimed to bridge the gap between theoretical equations and production-grade code.

**Key Objectives:**

1.  **Mathematical Rigor:** Implementing backpropagation, chain rule derivatives, and loss functions manually.
2.  **Performance Optimization:** Moving from naive scalar loops to **fully vectorized matrix operations** (Batch Processing) to significantly accelerate training times.
3.  **Software Architecture:** Applying **SOLID principles** to decouple Layers, Optimizers, and Loss functions for a modular design.

## Key Features

This library handles numerical stability and batch processing.

* **Vectorized Engine:** All forward and backward passes are optimized for mini-batch processing using NumPy broadcasting.
* **Numerical Stability:** Implementations of `Softmax` and `CrossEntropy` include log-sum-exp tricks to prevent underflow/overflow (NaNs) during training.
* **Modular Optimizers:** The optimization logic (SGD) is decoupled from the layer logic, allowing for easy extension to Adam or RMSprop.
* **Flexible Architecture:** Supports arbitrary sequences of Dense layers and Activation functions.

## Implemented Components

| Component | Details |
| :--- | :--- |
| **Layers** | `Dense` (Fully Connected), `Input`, `Reshape` |
| **Activations** | `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`, `PReLU` |
| **Loss Functions** | `MSE` (Mean Squared Error), `BinaryCrossEntropy`, `CategoricalCrossEntropy` (Logits optimized) |
| **Optimizers** | `SGD` (Stochastic Gradient Descent) |

## Installation

You can install the package directly from PyPI:

```bash
pip install mpneuralnetwork
````

Or clone the repository to work on the source code.

## Usage Example

The API is designed to be declarative and intuitive. Here is how to solve the classic MNIST digit classification problem:

```python
import numpy as np
from mpneuralnetwork.layers import Dense
from mpneuralnetwork.activations import ReLU, Softmax
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.model import Model

# 1. Define the Architecture
# Input: 784 (28x28 pixels) -> Hidden: 128 -> Output: 10 classes
network = [
    Dense(784, 128),
    ReLU(),
    Dense(128, 10),
    Softmax()
]

# 2. Initialize the Model
# We use Categorical Cross Entropy suited for multi-class classification
model = Model(
    network=network, 
    loss=CategoricalCrossEntropy()
)

# 3. Train (Vectorized)
# The model handles batching internally for performance
model.train(X_train, y_train, epochs=20, batch_size=32)

# 4. Predict
predictions = model.predict(X_test)
```

## Architecture & Design Decisions

### 1\. Vectorization & Performance

Early versions of the library used loops to iterate over samples one by one. This was identified as a major bottleneck.

  * **Refactoring:** I completely rewrote the main training loop (`Model.train`) and the `forward`/`backward` methods of all layers to handle 3D/2D tensors of shape `(batch_size, features)`.
  * **Result:** On the MNIST dataset, training time for 10 epochs dropped from **452s to 119s** (\~4x speedup).

### 2\. Decoupling Layers & Optimizers (SRP)

To avoid "God Classes", I strictly separated the responsibility of **calculating gradients** from **updating parameters**.

  * **The Layer's Job:** It computes `dE/dW` (gradient) during the backward pass and stores it in `self.weights_gradient`, but **it does not update its own weights**.
  * **The Optimizer's Job:** The `Optimizer` class (e.g., `SGD`) iterates over the layers, checks for the existence of gradients using introspection, and applies the update rule.

```python
# Simplified logic from optimizers.py
class SGD(Optimizer):
    def step(self, layers):
        for layer in layers:
            # The optimizer checks if the layer has trainable parameters
            if hasattr(layer, "weights_gradient"):
                # It applies the update rule externally
                layer.weights -= self.learning_rate * layer.weights_gradient
```

This architecture allows swapping `SGD` for `Adam` or `RMSprop` without changing a single line of code in the `Dense` or `Convolutional` layers.

## Roadmap

  * [x] **Batch Vectorization**
  * [x] **Numerical Stability Fixes (Logits)**
  * [ ] **Keras-like API:** Transitioning to `model.compile()` and `model.fit()` syntax for better standardisation.
  * [ ] **Convolutional Layers (CNN):** Implementation of `im2col` algorithm for efficient convolution.
  * [ ] **Advanced Optimizers:** Adam, RMSprop.
  * [ ] **Model Serialization:** Saving/Loading weights.

## Author

**Maxime Pires** - *AI Engineer | CentraleSupélec*

Building robust AI systems by understanding the foundations.

[LinkedIn](https://www.linkedin.com/in/maximepires/) | [Portfolio](https://github.com/maximepires4)
