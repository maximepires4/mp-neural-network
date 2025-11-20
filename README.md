# **MPNeuralNetwork 🧠**

**A fully vectorized Deep Learning framework built from scratch using only NumPy.**

## **Philosophy & Goal**

In an era of high-level frameworks like PyTorch, TensorFlow and Keras, it is easy to treat Neural Networks as "black boxes".

MPNeuralNetwork is an engineering initiative designed to demystify the underlying mathematics of Deep Learning.
By rebuilding the engine from the ground up, I aimed to bridge the gap between theoretical equations and production-grade code.  

## **Key Objectives:**

1. **Mathematical Rigor:** Implementing backpropagation, chain rule derivatives, and loss functions manually.  
2. **Performance Optimization:** Moving from naive scalar loops to **fully vectorized matrix operations** (Batch Processing) to significantly accelerate training times.  
3. **Software Architecture:** Applying **SOLID principles** to decouple Layers, Optimizers, and Loss functions for a modular design.

## **Key Features**

This library handles numerical stability, batch processing, and intelligent defaults.

* **Vectorized Engine:** All forward and backward passes are optimized for mini-batch processing using NumPy broadcasting.  
* **Smart Initialization:** The model automatically detects the activation function following a Dense layer and applies the optimal initialization strategy (**He** for ReLU, **Xavier** for Sigmoid/Tanh).  
* **Auto-Activation:** For classification tasks, the model automatically applies the correct output activation (Softmax for Multi-class, Sigmoid for Binary) during prediction, ensuring numerical stability during training (Logits).  
* **Modular Optimizers:** The optimization logic is strictly decoupled from the layer logic, allowing for stateful optimizers like Adam or RMSprop.

## **Implemented Components**

| Component | Details |
| :---- | :---- |
| **Layers** | Dense, Convolutional, Dropout, Reshape |
| **Activations** | ReLU, Sigmoid, Tanh, Softmax, PReLU, Swish |
| **Loss Functions** | MSE (Regression), BinaryCrossEntropy, CategoricalCrossEntropy (Logits optimized) |
| **Optimizers** | SGD (with Momentum), RMSprop, Adam |

## **Installation**

You can install the package directly from PyPI:

```
pip install mpneuralnetwork
```

Or clone the repository to work on the source code.

## **Usage Example**

The API is designed to be declarative and intuitive. Here is how to solve the classic MNIST digit classification problem using the "Smart API":

```
import numpy as np  
from mpneuralnetwork.layers import Dense, Dropout  
from mpneuralnetwork.activations import ReLU  
from mpneuralnetwork.losses import CategoricalCrossEntropy  
from mpneuralnetwork.optimizers import Adam  
from mpneuralnetwork.model import Model

# 1. Define the Architecture  
# Note: We use 'auto' initialization and NO final Softmax layer.  
network = [  
    Dense(784, 128, initialization='auto'), # Automatically uses He init  
    ReLU(),  
    Dropout(0.2),                           # Regularization  
    Dense(128, 10, initialization='auto')   # Output Logits  
]

# 2. Initialize the Model  
# The model detects CategoricalCrossEntropy and will automatically:  
# - Use Softmax for predictions  
# - Use raw Logits for stable training  
model = Model(  
    layers=network,   
    loss=CategoricalCrossEntropy(),  
    optimizer=Adam(learning_rate=0.001)  
)

# 3. Train (Vectorized)  
model.train(X_train, y_train, epochs=10, batch_size=32)

# 4. Predict  
# Returns probabilities (Softmax applied automatically)  
predictions = model.predict(X_test)
```

## **Architecture & Design Decisions**

### **1\. Vectorization & Performance**

Early versions of the library used loops to iterate over samples one by one. This was identified as a major bottleneck.

* **Refactoring:** I completely rewrote the main training loop (Model.train) and the forward/backward methods of all layers to handle 3D/2D tensors of shape (batch\_size, features).  
* **Result:** On the MNIST dataset, training time for 10 epochs dropped from **452s to 119s** (\~4x speedup).

### **2\. Decoupling Layers & Optimizers (SRP)**

To avoid "God Classes", I strictly separated the responsibility of **calculating gradients** from **updating parameters**. Layers expose their trainable parameters via a generic params property.

* **The Layer's Job:** It computes dE/dW (gradient) during the backward pass.  
* **The Optimizer's Job:** The Optimizer class iterates over the layers, retrieves parameters via layer.params, and applies the update rule (keeping track of momentum/velocity if needed).

```
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
* [ ] **Convolutional Optimization:** Implementation of im2col for faster CNNs.  
* [ ] **Model Serialization:** Saving/Loading weights to JSON/Pickle.

## **Author**

**Maxime Pires** - *AI Engineer | CentraleSupelec*

Building robust AI systems by understanding the foundations.

[LinkedIn](https://www.linkedin.com/in/maximepires) | [Portfolio](https://github.com/maximepires4)
