import numpy as np
import pytest

from mpneuralnetwork.activations import ReLU, Softmax
from mpneuralnetwork.layers import BatchNormalization, Dense
from mpneuralnetwork.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from mpneuralnetwork.metrics import Accuracy
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import SGD


def test_model_raises_error_if_input_size_missing():
    with pytest.raises(ValueError, match="Input layer does not define input size"):
        Model([Dense(10)], MSE())
    with pytest.raises(ValueError, match="Input layer does not define input size"):
        Model([BatchNormalization()], MSE())


def test_smart_weight_initialization():
    input_size = 1000
    output_size = 1000

    layers_he = [Dense(output_size, input_size=input_size, initialization="auto"), ReLU()]
    Model(layers_he, MSE(), SGD())
    weights_he = layers_he[0].weights
    expected_std_he = np.sqrt(2.0 / input_size)
    actual_std_he = np.std(weights_he)
    assert np.isclose(actual_std_he, expected_std_he, rtol=0.05)

    layers_xavier = [Dense(output_size, input_size=input_size, initialization="auto")]
    Model(layers_xavier, MSE(), SGD())
    weights_xavier = layers_xavier[0].weights
    expected_std_xavier = np.sqrt(1.0 / input_size)
    actual_std_xavier = np.std(weights_xavier)
    assert np.isclose(actual_std_xavier, expected_std_xavier, rtol=0.05)

    layers_forced = [Dense(output_size, input_size=input_size, initialization="xavier"), ReLU()]
    Model(layers_forced, MSE(), SGD())
    weights_forced = layers_forced[0].weights
    actual_std_forced = np.std(weights_forced)
    assert np.isclose(actual_std_forced, expected_std_xavier, rtol=0.05)


def test_model_duplicate_activation_removal():
    layers = [Dense(10, input_size=5), Softmax()]
    loss = CategoricalCrossEntropy()
    model = Model(layers, loss)
    assert len(model.layers) == 1
    assert isinstance(model.output_activation, Softmax)


def test_smart_metrics_initialization():
    model_reg = Model([Dense(1, input_size=1)], MSE())
    metric_names = [m.__class__.__name__ for m in model_reg.metrics]
    assert "RMSE" in metric_names
    assert "R2Score" in metric_names

    model_clf = Model([Dense(1, input_size=1)], BinaryCrossEntropy())
    metric_names = [m.__class__.__name__ for m in model_clf.metrics]
    assert "Accuracy" in metric_names
    assert "F1Score" in metric_names

    custom_metrics = [Accuracy()]
    model_custom = Model([Dense(1, input_size=1)], MSE(), metrics=custom_metrics)
    assert len(model_custom.metrics) == 1
    assert isinstance(model_custom.metrics[0], Accuracy)
