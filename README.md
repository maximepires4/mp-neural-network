# MPNeuralNetwork

Custom library for implementing feedforward neural networks using back propagation.
I recreated the algorithm from scratch, without using any external library, to fully understand the process.

## Features

This library provides functions to create and run neural networks.

* Layers : Dense
* Activations : Tanh, Sigmoid, ReLU, PReLU, Softmax
* Losses : MSE, Cross-Entropy

## Usage

Create a network using layers and activations classes. Example:

```python3
network = [Dense(784, 128), Tanh(), Dense(128, 40), Tanh(), Dense(40, 10), Softmax()]
```

Create a model using the network previously created and a loss class. Example:

```python3
model = Model(network, MSE())
```

Train the model by setting some parameters. Example:

```python3
model.train(input, output, epochs=10, learning_rate=0.1, batch_size=10)
```

## Example

See my [handwriting recognition repository](https://github.com/maximepires4/handwriting-recognition).
