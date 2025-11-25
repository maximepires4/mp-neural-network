import sys
from io import StringIO

import numpy as np

from mpneuralnetwork.layers import Dense
from mpneuralnetwork.losses import MSE
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import SGD


def test_retraining_maintains_optimizer_state():
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randn(50, 1)

    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model = Model([Dense(1, input_size=5)], MSE(), optimizer)

    model.train(X, y, epochs=1, batch_size=10)

    assert len(optimizer.velocities) > 0
    param_id = list(optimizer.velocities.keys())[0]
    velocity_after_epoch1 = optimizer.velocities[param_id].copy()
    assert not np.allclose(velocity_after_epoch1, 0)

    model.train(X, y, epochs=1, batch_size=10)

    velocity_after_epoch2 = optimizer.velocities[param_id]
    assert not np.allclose(velocity_after_epoch1, velocity_after_epoch2)

    loss_after = model.loss.direct(model.predict(X), y)
    assert loss_after < 10.0


def test_auto_evaluation_splits_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100, 1)

    model = Model([Dense(1, input_size=5)], MSE(), SGD(learning_rate=0.01))

    captured_output = StringIO()
    sys.stdout = captured_output

    try:
        model.train(X, y, epochs=1, batch_size=10, auto_evaluation=0.2)
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    assert "evaluation" in output
