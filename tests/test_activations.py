import numpy as np
import pytest
from mpneuralnetwork.activations import Sigmoid, Tanh, ReLU

# This file contains tests for the activation functions.
# These tests are fundamental as they validate the basic mathematical
# building blocks of the network, which will not change even after
# the planned refactoring.

def test_sigmoid():
    """
    Tests the Sigmoid activation function and its derivative.
    """
    sigmoid = Sigmoid()
    
    # Test the forward() method
    # sigmoid(0) must be 0.5
    assert np.isclose(sigmoid.forward(0), 0.5)
    # sigmoid(x) for x > 0 must be > 0.5
    assert sigmoid.forward(1) > 0.5
    # sigmoid(x) for x < 0 must be < 0.5
    assert sigmoid.forward(-1) < 0.5

    # Test the backward() method (the derivative)
    # The derivative is calculated based on the input from the last forward pass.
    # The maximum derivative is at x=0, and its value is 0.25
    sigmoid.forward(0)
    assert np.isclose(sigmoid.backward(1), 0.25)
    
    # The derivative is symmetric
    sigmoid.forward(1)
    derivative_at_1 = sigmoid.backward(1)
    sigmoid.forward(-1)
    derivative_at_minus_1 = sigmoid.backward(1)
    assert np.isclose(derivative_at_1, derivative_at_minus_1)
    
    # The derivative must be positive
    sigmoid.forward(2)
    assert sigmoid.backward(1) > 0

def test_tanh():
    """
    Tests the Tanh activation function and its derivative.
    """
    tanh = Tanh()

    # Test the forward() method
    assert np.isclose(tanh.forward(0), 0)
    assert tanh.forward(1) > 0
    assert tanh.forward(-1) < 0
    
    # Test the backward() method
    # The maximum derivative is at x=0, and its value is 1
    tanh.forward(0)
    assert np.isclose(tanh.backward(1), 1)

    # The derivative is symmetric
    tanh.forward(2)
    derivative_at_2 = tanh.backward(1)
    tanh.forward(-2)
    derivative_at_minus_2 = tanh.backward(1)
    assert np.isclose(derivative_at_2, derivative_at_minus_2)

def test_relu():
    """
    Tests the ReLU activation function and its derivative.
    """
    relu = ReLU()

    # Test the forward() method
    assert np.isclose(relu.forward(0), 0)
    assert np.isclose(relu.forward(10), 10)
    assert np.isclose(relu.forward(-10), 0)

    # Test the backward() method
    relu.forward(10)
    assert np.isclose(relu.backward(1), 1)
    
    relu.forward(-10)
    assert np.isclose(relu.backward(1), 0)
    
    # The derivative at 0 is 0 in this implementation
    relu.forward(0)
    assert np.isclose(relu.backward(1), 0)