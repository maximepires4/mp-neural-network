import gzip
from pathlib import Path

import dill as pickle
import numpy as np

from mpneuralnetwork.activations import Softmax, Tanh
from mpneuralnetwork.layers import Convolutional, Dense, Reshape
from mpneuralnetwork.losses import MSE
from mpneuralnetwork.model import Model


def load_data():
    with gzip.open("data/mnist.pkl.gz", "rb") as f:
        f.seek(0)
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
        return (training_data, validation_data, test_data)


training_data, validation_data, test_data = load_data()

input = training_data[0]

output = np.zeros((training_data[1].shape[0], 10))
for i in range(training_data[1].shape[0]):
    output[i, training_data[1][i]] = 1

input = input.reshape((50000, 1, 28, 28))
output = output.reshape((50000, 10, 1))

network = [
    Convolutional((1, 28, 28), 3, 5),
    Tanh(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Tanh(),
    Dense(5 * 26 * 26, 100),
    Tanh(),
    Dense(100, 10),
    Softmax(),
]

model = Model(network, MSE())

model.train(input, output, epochs=10, batch_size=10)

Path("output/").mkdir(parents=True, exist_ok=True)

with open("output/model-convolutional.pkl", "wb") as f:
    pickle.dump(model, f)
