# User Guide: Getting Started

This tutorial will walk you through the essential concepts of the framework, from installation to building your first Deep Learning models.

---

## 1. Installation

First, ensure you have Python 3.10 or later installed.

### From PyPI (Stable)

```bash
pip install mpneuralnetwork
```

### From Source (Development)

```bash
git clone https://github.com/maximepires4/mp-neural-network.git
cd mp-neural-network
pip install -e .
```

---

## 2. Hardware Acceleration (GPU)

MPNeuralNetwork supports GPU acceleration using **CuPy**.

### Requirements

* NVIDIA GPU
* CUDA Toolkit installed
* `cupy` installed (`pip install cupy-cuda12x` depending on your CUDA version)

### Usage

To enable GPU mode, simply set the `MPNN_BACKEND` environment variable before running your script:

```bash
# Linux / macOS
export MPNN_BACKEND=cupy
python my_script.py
```

If CuPy is not found or the variable is not set, it falls back to NumPy (CPU).

---

## 3. Core Concepts

MPNeuralNetwork exposes some of the mathematical machinery while keeping the API clean.

### The `Model` Container

Everything revolves around the `Model` class. It manages:

1. **Layers**: The sequence of transformations (Dense, Conv2D, etc.).
2. **Loss Function**: How error is calculated (MSE, CrossEntropy).
3. **Optimizer**: How weights are updated (SGD, Adam).

### Data Format

The framework uses **NumPy arrays** (or CuPy if on GPU).

* **Inputs (X):** Must be float32 arrays of shape `(batch_size, features...)`.
* **Targets (y):**
  * **Regression:** `(batch_size, outputs)`.
  * **Classification:** One-hot encoded `(batch_size, num_classes)`.

---

## 3. Tutorial 1: Your First Neural Network (MNIST)

Let's build a classic Multi-Layer Perceptron (MLP) to classify handwritten digits from the MNIST dataset.

### Step 1: Prepare Data

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

# 1. Load Data
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy() / 255.0  # Normalize to [0, 1]
y = mnist.target.to_numpy().reshape(-1, 1)

# 2. One-Hot Encode Labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# 3. Split Train/Test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```

### Step 2: Define Architecture

We will build a network with:

* Input: 784 features (28x28 pixels flattened)
* Hidden Layer 1: 128 neurons + ReLU + Dropout
* Output Layer: 10 neurons (digits 0-9)

```python
from mpneuralnetwork.layers import Dense, Dropout
from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.optimizers import Adam
from mpneuralnetwork.model import Model

network = [
    Dense(128, input_size=784),  # Weights auto-initialized (He Init)
    ReLU(),
    Dropout(0.2),                # 20% dropout during training
    Dense(10)                    # Output logits (no Softmax here!)
]

model = Model(
    layers=network,
    loss=CategoricalCrossEntropy(),  # Handles Softmax internally
    optimizer=Adam(learning_rate=0.001),
    metrics=[] # Accuracy is auto-added
)
```

### Step 3: Train

We use the `train()` method. Note the `auto_evaluation` parameter which automatically creates a validation set.

```python
model.train(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    auto_evaluation=0.2,  # Use 20% of data for validation
    early_stopping=3      # Stop if no improvement for 3 epochs
)
```

### Step 4: Evaluate & Predict

```python
# Check performance on test set
model.test(X_test, y_test)

# Make predictions
preds = model.predict(X_test[:5])
print("Predicted probabilities:", preds)
```

---

## 4. Tutorial 2: Convolutional Neural Network (CNN)

For image data, CNNs are superior. MPNeuralNetwork supports 2D convolutions using the optimized `im2col` technique.

**Note on Shapes:** CNNs expect 4D input tensors: `(Batch, Channels, Height, Width)`.

```python
# Reshape flat MNIST data to 4D
X_train_cnn = X_train.reshape(-1, 1, 28, 28)
X_test_cnn = X_test.reshape(-1, 1, 28, 28)
```

### Defining the CNN

```python
from mpneuralnetwork.layers import Convolutional, Flatten, MaxPooling2D, BatchNormalization2D

cnn_network = [
    # Conv1: 1 input channel -> 32 filters of size 3x3
    # stride=1 and padding=1 preserves spatial dimensions (28x28)
    Convolutional(output_depth=32, kernel_size=3, input_shape=(1, 28, 28), stride=1, padding=1),
    BatchNormalization2D(),
    ReLU(),

    # Pool1: Downsample by 2 (14x14)
    MaxPooling2D(pool_size=2, strides=2),

    # Conv2: 32 input channels -> 64 filters
    Convolutional(output_depth=64, kernel_size=3, stride=1, padding=1),
    BatchNormalization2D(),
    ReLU(),

    # Pool2: Downsample by 2 (7x7)
    MaxPooling2D(pool_size=2, strides=2),

    # Flatten: Convert 3D feature maps to 1D vector
    Flatten(),

    # Dense Classifier
    Dense(100),
    ReLU(),
    Dense(10)
]

model_cnn = Model(
    layers=cnn_network,
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001)
)

model_cnn.train(X_train_cnn, y_train, epochs=10, batch_size=32)
```

---

## 5. Saving and Loading

You can save your trained model to disk to use it later without retraining.

```python
from mpneuralnetwork.serialization import save_model, load_model

# Save full model (architecture + weights + optimizer state)
save_model(model, "my_mnist_model.npz")

# Load it back
loaded_model = load_model("my_mnist_model.npz")

# Resume training or predict immediately
loaded_model.predict(X_test[:1])
```

---

## 6. Next Steps

* Learn about performance tuning in the [Optimization Guide](OPTIMIZATION_GUIDE.md).
* Explore the full [API Reference](reference/model.md).
