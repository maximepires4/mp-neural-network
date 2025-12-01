from abc import abstractmethod
from typing import Literal

import numpy as np

from .. import ArrayType

Lit_W = Literal["auto", "he", "xavier"]


class Layer:
    """Abstract base class for all neural network layers.

    This class defines the interface that all layers must implement, including
    forward/backward passes and parameter management.

    Attributes:
        input_shape (tuple[int, ...]): Shape of the input data (excluding batch dimension).
        output_shape (tuple[int, ...]): Shape of the output data (excluding batch dimension).
        input (ArrayType): Caches the input for the backward pass.
        output (ArrayType): Caches the output.
    """

    def __init__(self, output_shape: int | tuple[int, ...] | None = None, input_shape: int | tuple[int, ...] | None = None) -> None:
        """Initializes the Layer.

        Args:
            output_shape (int | tuple[int, ...], optional): Desired output shape.
            input_shape (int | tuple[int, ...], optional): Known input shape.
        """
        self.output_shape: tuple[int, ...]
        if output_shape is not None:
            if isinstance(output_shape, int):
                output_shape = (output_shape,)
            self.output_shape = output_shape

        self.input_shape: tuple[int, ...]
        if input_shape is not None:
            if isinstance(input_shape, int):
                input_shape = (input_shape,)
            self.input_shape = input_shape

        self.input: ArrayType
        self.output: ArrayType

    def get_config(self) -> dict:
        """Returns the configuration of the layer for serialization.

        Returns:
            dict: Dictionary containing layer configuration.
        """
        return {"type": self.__class__.__name__}

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        """Configures the layer based on the input shape.

        Called automatically by the Model before training.

        Args:
            input_shape (int | tuple[int, ...]): The shape of the input.
        """
        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        self.input_shape = input_shape

        if not hasattr(self, "output_shape"):
            self.output_shape = input_shape

    @abstractmethod
    def forward(self, input_batch: ArrayType, training: bool = True) -> ArrayType:
        """Performs the forward propagation pass.

        Args:
            input_batch (ArrayType): Input data of shape (batch_size, ...).
            training (bool, optional): Whether the layer is in training mode. Defaults to True.

        Returns:
            ArrayType: Output data.
        """
        pass

    @abstractmethod
    def backward(self, output_gradient_batch: ArrayType) -> ArrayType:
        """Performs the backward propagation pass.

        Computes the gradient of the loss function with respect to the input.
        Also calculates gradients for any trainable parameters.

        Args:
            output_gradient_batch (ArrayType): Gradient of the loss w.r.t the output.

        Returns:
            ArrayType: Gradient of the loss w.r.t the input.
        """
        pass

    @property
    def params(self) -> dict[str, tuple[ArrayType, ArrayType]]:
        """Returns trainable parameters and their gradients.

        Returns:
            dict[str, tuple[ArrayType, ArrayType]]: A dictionary where keys are parameter names
            (e.g., "weights", "biases") and values are tuples of (parameter_value, parameter_gradient).
        """
        return {}

    def load_params(self, params: dict[str, ArrayType]) -> None:
        """Loads trainable parameters into the layer.

        Args:
            params (dict[str, ArrayType]): Dictionary mapping parameter names to values.
        """
        pass

    @property
    def state(self) -> dict[str, ArrayType]:
        """Returns non-trainable internal state (e.g., BatchNorm running means).

        Returns:
            dict[str, ArrayType]: Dictionary of state variables.
        """
        return {}

    @state.setter
    def state(self, state: dict[str, ArrayType]) -> None:
        """Restores non-trainable internal state.

        Args:
            state (dict[str, ArrayType]): Dictionary of state variables.
        """
        pass

    @property
    def input_size(self) -> int:
        """Returns the total number of elements in the input (excluding batch)."""
        return int(np.prod(self.input_shape))

    @property
    def output_size(self) -> int:
        """Returns the total number of elements in the output (excluding batch)."""
        return int(np.prod(self.output_shape))
