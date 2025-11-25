from collections.abc import Iterator

import numpy as np
import pytest


class MockTrainableLayer:
    """
    A mock layer with trainable parameters that exposes them via a `params` property.
    """

    def __init__(self) -> None:
        self.weights = np.ones((10, 5))
        self.biases = np.ones((1, 5))
        self.weights_gradient = np.full_like(self.weights, 0.5)
        self.biases_gradient = np.full_like(self.biases, 0.2)

    @property
    def params(self) -> dict:
        """Exposes parameters and their corresponding gradients."""
        return {
            "weights": (self.weights, self.weights_gradient),
            "biases": (self.biases, self.biases_gradient),
        }


class MockNonTrainableLayer:
    """A mock layer with no trainable parameters."""

    pass


@pytest.fixture
def mock_trainable_layer() -> Iterator[MockTrainableLayer]:
    """Fixture that returns a fresh MockTrainableLayer instance."""
    yield MockTrainableLayer()
