# Layers API

Layers are the building blocks of neural networks. They store parameters (weights, biases) and implement the forward/backward propagation logic.

## Base Layer

::: mpneuralnetwork.layers.Layer

## 1D Layers (Dense)

Layers typically used for Multi-Layer Perceptrons (MLP) or final classification stages.

::: mpneuralnetwork.layers.Dense
::: mpneuralnetwork.layers.Dropout
::: mpneuralnetwork.layers.BatchNormalization

## 2D Layers (Convolutional)

Layers designed for processing grid-like data (e.g., images) using the `im2col` optimization.

::: mpneuralnetwork.layers.Convolutional
::: mpneuralnetwork.layers.MaxPooling2D
::: mpneuralnetwork.layers.AveragePooling2D
::: mpneuralnetwork.layers.Flatten
::: mpneuralnetwork.layers.BatchNormalization2D

## Utilities

Low-level utility functions used for implementing efficient convolutions.

::: mpneuralnetwork.layers.utils.im2col
::: mpneuralnetwork.layers.utils.col2im
