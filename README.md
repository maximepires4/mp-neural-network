<p align="center">
  <img src="images/logo.svg" alt="MPNeuralNetwork Logo" width="900"/>
</p>

# **MPNeuralNetwork 🧠**

[![PyPI version](https://badge.fury.io/py/mpneuralnetwork.svg)](https://badge.fury.io/py/mpneuralnetwork)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A fully vectorized Deep Learning framework built from scratch using only NumPy.**

## **Philosophy & Goal**

In an era of high-level frameworks like PyTorch, TensorFlow and Keras, it is easy to treat Neural Networks as "black boxes".

MPNeuralNetwork is an engineering initiative designed to demystify the underlying mathematics of Deep Learning.
By rebuilding the engine from the ground up, I aimed to bridge the gap between theoretical equations and production-grade code.

## **Key Objectives:**

1. **Mathematical Rigor:** Implementing backpropagation, chain rule derivatives, and loss functions manually.
2. **Performance Optimization:** Moving from naive scalar loops to **fully vectorized matrix operations** (Batch Processing) to significantly accelerate training times.
3. **Software Architecture:** Applying **SOLID principles** to decouple Layers, Optimizers, and Loss functions for a modular design.

## **Key Features: Smart & Efficient**

MPNeuralNetwork goes beyond basic matrix operations by incorporating an **"intelligent" engine** that automates Deep Learning best practices.

* **Fully Vectorized:** Optimized for batch processing using NumPy broadcasting for maximum performance.
* **Early Stopping & Checkpointing:** The training loop automatically monitors validation performance. It stops training early if the model stops learning and **automatically restores the best weights** found during training, ensuring you always get the most generalized model.
* **Intelligent Weight Initialization:** The model analyzes your network architecture (specifically the activation functions) and automatically applies the optimal initialization strategy (**He Initialization** for ReLU, **Xavier/Glorot** for Sigmoid/Tanh), removing the guesswork.
* **Numerical Stability (Auto-Logits):** The framework detects classification tasks and internally handles logits for `Softmax` or `Sigmoid`, preventing numerical overflow/underflow issues common in naive implementations.
* **Auto-Validation Split:** Simply pass `auto_evaluation=0.2` to automatically set aside 20% of your data for validation, without manual array slicing.
* **Full Serialization:** Save and load your entire model state (weights, architecture, optimizer momentum) to resume training later.

## **Implemented Components**

| Component | Details |
| :---- | :---- |
| **Layers** | `Dense`, `Convolutional` (Conv2D), `Dropout`, `Reshape`, `BatchNormalization` |
| **Activations** | `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `PReLU`, `Swish` |
| **Loss Functions** | `MSE` (Regression), `BinaryCrossEntropy`, `CategoricalCrossEntropy` (Logits optimized) |
| **Optimizers** | `SGD` (with Momentum), `RMSprop`, `Adam` |

## **Installation**

You can install the package directly from PyPI:

```bash
pip install mpneuralnetwork
```

Or clone the repository to work on the source code.

## **Usage Examples**

### **1. Classic MNIST Classification (MLP)**

The API is designed to be declarative and intuitive.

```python
import numpy as np
from mpneuralnetwork.layers import Dense, Dropout
from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.optimizers import Adam
from mpneuralnetwork.model import Model

# 1. Define the Architecture
# Note: We use 'auto' initialization and NO final Softmax layer (handled by loss).
network = [
    Dense(784, 128, initialization='auto'), # Automatically uses He init
    ReLU(),
    Dropout(0.2),                           # Regularization
    Dense(128, 10, initialization='auto')   # Output Logits
]

# 2. Initialize the Model
model = Model(
    layers=network,
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001)
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

### **2. Convolutional Neural Network (CNN)**

Support for 2D Convolutions for image processing tasks.

```python
from mpneuralnetwork.layers import Convolutional, Reshape, Dense
from mpneuralnetwork.activations import ReLU, Softmax

cnn_network = [
    # Input: (Batch, 1, 28, 28) -> Output: (Batch, 32, 26, 26)
    Convolutional(input_shape=(1, 28, 28), kernels_count=32, kernel_size=3),
    ReLU(),

    # Flatten: (Batch, 32 * 26 * 26)
    Reshape((-1, 32 * 26 * 26)),

    Dense(32 * 26 * 26, 100),
    ReLU(),
    Dense(100, 10)
]

model = Model(layers=cnn_network, ...)
```

### **3. Saving & Loading Models**

You can save the entire model state (weights, architecture, optimizer config) and reload it later.

```python
# Save
model.save("my_model") # Creates my_model.npz

# Load
loaded_model = Model.load("my_model")
loaded_model.predict(X_test)
```

## **Architecture & Design Decisions**

### **1\. Vectorization & Performance**

Early versions of the library used loops to iterate over samples one by one. This was identified as a major bottleneck.

* **Refactoring:** I completely rewrote the main training loop (`Model.train`) and the forward/backward methods of all layers to handle 3D/2D tensors of shape `(batch_size, features)`.
* **Result:** On the MNIST dataset, training time for 10 epochs dropped from **452s to 119s** (~4x speedup).

### **2\. Decoupling Layers & Optimizers (SRP)**

To avoid "God Classes", I strictly separated the responsibility of **calculating gradients** from **updating parameters**. Layers expose their trainable parameters via a generic params property.

* **The Layer's Job:** It computes `dE/dW` (gradient) during the backward pass.
* **The Optimizer's Job:** The Optimizer class iterates over the layers, retrieves parameters via `layer.params`, and applies the update rule (keeping track of momentum/velocity if needed).

```python
# Simplified logic from optimizers.py
class SGD(Optimizer):
    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, 'params'): continue

            # We use id(param) to track states (velocity) for specific weights
            for _, (param, grad) in layer.params.items():
                # Update logic...
                param -= self.learning_rate * grad
```

## **Roadmap**

* [x] **Batch Vectorization**
* [x] **Numerical Stability Fixes (Logits)**
* [x] **Advanced Optimizers:** Adam, RMSprop, SGD Momentum.
* [x] **Smart Initialization:** Auto He/Xavier.
* [x] **Regularization:** Dropout Layer.
* [x] **Convolutional Layers:** Conv2D implementation.
* [x] **Model Serialization:** Saving/Loading weights to JSON/Pickle.
* [x] **Training Utils:** Early Stopping, Checkpointing.
* [ ] **Pooling Layers:** MaxPool / AvgPool.
* [ ] **Convolutional Optimization:** Implementation of `im2col` for faster CNNs.

## **Author**

**Maxime Pires** - *AI Engineer | CentraleSupelec*

Building robust AI systems by understanding the foundations.

[LinkedIn](https://www.linkedin.com/in/maximepires) | [Portfolio](https://github.com/maximepires4)
