from abc import abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Lit_W = Literal["auto", "he", "xavier"]


class Layer:
    def __init__(self, output_shape: int | tuple[int, ...] | None = None, input_shape: int | tuple[int, ...] | None = None) -> None:
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

        self.input: NDArray
        self.output: NDArray

    def get_config(self) -> dict:
        return {"type": self.__class__.__name__}

    def build(self, input_shape: int | tuple[int, ...]) -> None:
        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        self.input_shape = input_shape

        if not hasattr(self, "output_shape"):
            self.output_shape = input_shape

    @abstractmethod
    def forward(self, input_batch: NDArray, training: bool = True) -> NDArray:
        pass

    @abstractmethod
    def backward(self, output_gradient_batch: NDArray) -> NDArray:
        pass

    @property
    def params(self) -> dict[str, tuple[NDArray, NDArray]]:
        return {}

    def load_params(self, params: dict[str, NDArray]) -> None:
        pass

    @property
    def state(self) -> dict[str, NDArray]:
        return {}

    @state.setter
    def state(self, state: dict[str, NDArray]) -> None:
        pass

    @property
    def input_size(self) -> int:
        return int(np.prod(self.input_shape))

    @property
    def output_size(self) -> int:
        return int(np.prod(self.output_shape))
