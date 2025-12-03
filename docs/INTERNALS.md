# Architecture & Internals

MPNeuralNetwork is not just a collection of matrix operations. It includes a lightweight "engine" that automates many of the tedious and error-prone aspects of Deep Learning configuration, built upon a modular architecture.

This guide explains the design decisions and the "magic" happening under the hood.

---

## 1. Design Philosophy (SOLID)

### Decoupling Layers & Optimizers (SRP)

To avoid "God Classes", the responsibility of **calculating gradients** from **updating parameters** is strictly separated.

* **The Layer's Job:** It computes `dE/dW` (gradient) during the backward pass and stores it. It knows *nothing* about learning rates or update rules.
* **The Optimizer's Job:** The Optimizer class iterates over the layers, retrieves parameters via the generic `layer.params` property, and applies the update rule (keeping track of momentum/velocity if needed).

### Unified Optimizer Design (Adam vs AdamW)

A common confusion in Deep Learning libraries is the difference between **Adam + L2 Regularization** and **AdamW** (Decoupled Weight Decay).

* **Standard Approach:** Usually, L2 regularization is added to the loss function, meaning its gradient is added to the backpropagated gradient. In adaptive methods like Adam, this means the regularization term is effectively scaled by the inverse of the gradient variance, which can be suboptimal.
* **MPNN Implementation:**
  * **L1 Regularization** follows the standard approach (added to gradients) to promote sparsity.
  * **L2 Regularization** is implemented as **Decoupled Weight Decay (AdamW)**. The decay is applied directly to the weights *after* the adaptive update step.

This design means that by simply setting `optimizer=Adam(regularization='L2')`, you are effectively using **AdamW**, the state-of-the-art optimizer for training modern deep networks.

---

## 2. Smart Weight Initialization

One of the most common reasons for a neural network failing to converge is improper weight initialization.

### The Problem

* **Sigmoid/Tanh** activations require weights to be small to avoid saturation (vanishing gradients).
* **ReLU** units require slightly larger weights to maintain variance (avoiding dead neurons).

### The Solution

The `Model` class performs a **static analysis** of your architecture before training begins. It looks at the connection between layers to decide the optimal initialization strategy.

* **He Initialization (Kaiming):** Applied if the layer is followed by `ReLU`, `PReLU`, or `Swish`.
* **Xavier Initialization (Glorot):** Applied if the layer is followed by `Sigmoid`, `Tanh`, or `Softmax`.
* **Bias Disabling:** If a `Dense` or `Convolutional` layer is immediately followed by `BatchNormalization`, the bias of the preceding layer is automatically disabled (set to `False`), as normalization makes it redundant.

```python
# Example of implicit automation
model = Model([
    Dense(100, input_size=784),  # Engine detects ReLU next -> Uses He Init
    ReLU(),
    Dense(10)                    # Engine detects nothing/Softmax next -> Uses Xavier
])
```

---

## 3. Training Loop Automation

The `Model.train()` method encapsulates a production-grade training loop with several automated features.

### Auto-Validation Split

Instead of manually slicing your NumPy arrays, you can pass `auto_evaluation=0.2`.

* The engine shuffles the data.
* It reserves the last 20% for validation.
* It ensures no data leakage between training and validation sets.

### Early Stopping & Checkpointing

The model monitors the validation loss at every epoch.

* **Patience:** If the loss doesn't improve for `early_stopping` epochs, training halts to prevent overfitting.
* **Best Weight Restoration:** Crucially, the model keeps a copy of the weights that achieved the *lowest validation loss*. When training ends (naturally or via early stopping), these "best weights" are automatically restored. You never end up with the overfitted weights from the final epoch.

---

## 4. Backend Abstraction (CPU/GPU)

To support both CPU and GPU execution without code duplication, MPNeuralNetwork implements a unified backend interface.

### The `xp` Abstraction

The library defines a global `xp` module alias that points to either:

* `numpy` (for CPU execution)
* `cupy` (for NVIDIA GPU execution)

All tensor operations (creation, math, reshaping) use `xp.array`, `xp.dot`, etc., instead of hardcoded `np.*` calls.

### Device Transfers

The `model` and `data` must reside on the same device.

* `to_device(array)`: Moves a NumPy array to the configured device (GPU if enabled).
* `to_host(array)`: Moves a device array back to CPU (NumPy) for printing or saving.

---

## 5. Backend Optimizations

### Vectorization & `im2col`

Python loops are too slow for Deep Learning. MPNN vectorizes operations to leverage BLAS/LAPACK routines via NumPy.

* **Batch Processing:** All layers operate on 3D/4D tensors `(Batch_Size, ...)`, eliminating the outer loop over samples.
* **Convolution via `im2col`:**
  * 2D Convolutions are difficult to vectorize directly.
  * **Solution:** implementation of `im2col` (Image to Column), which stretches image patches into columns.
  * This transforms the convolution operation into a single, massive Matrix Multiplication (`GEMM`).
  * **Result:** Orders of magnitude faster than iterative sliding windows.

### Memory Management (Float32)

By default, Python and NumPy use `float64` (double precision). Deep Learning rarely needs this precision.

* The framework enforces `DTYPE = np.float32` globally.
* Inputs are automatically cast to float32.
* This reduces RAM usage by **50%** and doubles memory bandwidth throughput.

---

## 5. Numerical Stability

### Logits Handling

Calculating `Softmax` then `CrossEntropy` separately is numerically unstable (can lead to `NaN` or `Infinity` due to exponentials).

* MPNN uses the "Logits" pattern.
* The `CategoricalCrossEntropy` and `BinaryCrossEntropy` losses expect raw outputs (logits) from the final layer, not probabilities.
* They internally compute the loss using the log-sum-exp trick or equivalent stable formulas.

*Note: When you call `model.predict()`, the engine automatically applies the final activation (Softmax/Sigmoid) so you get human-readable probabilities.*

## 6. Smart Metrics

To reduce boilerplate, the framework automatically assigns relevant metrics if the user doesn't specify any.

* **Regression (MSE):** Automatically tracks `RMSE` and `R2Score`.
* **Classification (CrossEntropy):** Automatically tracks `Accuracy` and `F1Score`.
