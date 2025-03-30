import gzip, pickle
import numpy as np

import dill as pickle
from pathlib import Path

from src.neuralnetwork.activations import Tanh, Sigmoid, Softmax, ReLU, PReLU, Swish
from src.neuralnetwork.losses import MSE
from src.neuralnetwork.layers import Dense
from src.neuralnetwork.model import Model

def load_data():
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        f.seek(0)
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        return (training_data, validation_data, test_data)
    
training_data, validation_data, test_data = load_data()

input = training_data[0]

output = np.zeros((training_data[1].shape[0], 10))
for i in range(training_data[1].shape[0]):
    output[i, training_data[1][i]] = 1

input = input.reshape((50000, 784, 1))
output = output.reshape((50000, 10, 1))

network = [
    Dense(784, 128),
    Tanh(),
    Dense(128, 40),
    Tanh(),
    Dense(40, 10),
    Softmax()
]

model = Model(network, MSE())

model.train(input, output, epochs=20, learning_rate=0.1, batch_size=10)

Path("output/").mkdir(parents=True, exist_ok=True)

with open('output/model.pkl', 'wb') as f:
    pickle.dump(model, f)