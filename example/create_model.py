import gzip
import random
from pathlib import Path

import dill as pickle
import numpy as np

from mpneuralnetwork.activations import Tanh
from mpneuralnetwork.layers import Dense, Dropout
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.model import Model


def load_data():
    with gzip.open("data/mnist.pkl.gz", "rb") as f:
        f.seek(0)
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
        return (training_data, validation_data, test_data)


seed = 69
np.random.seed(seed)
random.seed(seed)

training_data, validation_data, test_data = load_data()

input = training_data[0]

output = np.zeros((training_data[1].shape[0], 10))
for i in range(training_data[1].shape[0]):
    output[i, training_data[1][i]] = 1

network = [Dense(128, input_size=784), Tanh(), Dropout(0.2), Dense(40), Tanh(), Dense(10)]

model = Model(network, CategoricalCrossEntropy())

model.train(input, output, epochs=10, batch_size=10)

Path("output/").mkdir(parents=True, exist_ok=True)

with open("output/model.pkl", "wb") as f:
    pickle.dump(model, f)
